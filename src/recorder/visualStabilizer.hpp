#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video.hpp>

#include "../common/logger.hpp"
#include "../math/support.hpp"


#ifndef OPTONAUT_VISUAL_STABILIZER_HEADER
#define OPTONAUT_VISUAL_STABILIZER_HEADER

using namespace std;
using namespace cv;

namespace optonaut {
class VisualStabilizer {
    private: 

    static const bool debug = false;

    double translationToRotation(double t, double pxpm, double r, double theta) {
       auto mm = t / pxpm;  
       auto a = atan2(mm, r * (1 + sin(theta)));
       return -a;
    }

    Mat prevExtrinsics;
    Mat sensorEstimate;
    InputImageP prevImage;

    public: 

    Mat GetCurrentEstimate() {
        return sensorEstimate;
    }

    void Push(const InputImageP &in) {
       if(prevExtrinsics.cols == 0) {
            // Init case
            in->originalExtrinsics.copyTo(prevExtrinsics);
            in->originalExtrinsics.copyTo(sensorEstimate);
            prevImage = in;
       } else {
            // Update case

           auto sensorDiff = prevExtrinsics.inv() * in->originalExtrinsics;
           STimer transform;
           auto estimatedDiff = estimateRigidTransform(prevImage->image.data, in->image.data, false);
           transform.Tick("transform-estimation");

           // Assuming principal point is in PX
           auto pxpm = in->image.cols / (in->intrinsics.at<double>(0, 2) * 2);
           auto r = 6;

           Mat sensorDiffVec;
           ExtractRotationVector(sensorDiff, sensorDiffVec);
           double theta = GetDistanceY(in->originalExtrinsics, Mat::eye(4, 4, CV_64F));

           if(debug) {
               double phi = GetDistanceX(in->originalExtrinsics, Mat::eye(4, 4, CV_64F));

               LogR << "t=" << in->id 
                  << " type=sensor-diff-x"
                  << " val=" << sensorDiffVec.at<double>(0);
               LogR << "t=" << in->id
                  << " type=sensor-diff-y"
                  << " val=" << sensorDiffVec.at<double>(1);
               LogR << "t=" << in->id
                  << " type=sensor-diff-z"
                  << " val=" << sensorDiffVec.at<double>(2);
               LogR << "t=" << in->id
                  << " type=sensor-abs-x"
                  << " val=" << phi;
               LogR << "t=" << in->id
                  << " type=sensor-abs-y"
                  << " val=" << theta;
           }

           if(estimatedDiff.cols != 0) {
               double estPhiDiff = translationToRotation(estimatedDiff.at<double>(0, 2), pxpm, r, theta);
               double phiDiff = sensorDiffVec.at<double>(1);
               double resudialPhiDiff = phiDiff - estPhiDiff;

               if(debug) {
                   double estThetaDiff = translationToRotation(estimatedDiff.at<double>(1, 2), pxpm, r, theta);
                   LogR << "t=" << in->id
                      << " type=estimated-diff-x"
                      << " val=" << estPhiDiff;
                   LogR << "t=" << in->id 
                      << " type=estimated-diff-y"
                      << " val=" << estThetaDiff; 
                   LogR << "t=" << in->id
                      << " type=resudial-diff-y"
                      << " val=" << resudialPhiDiff;
               }

               // If we derive by too much, and video is smaller than sensor, use video. 
               if(std::abs(resudialPhiDiff) > 0.05 && std::abs(estPhiDiff) < std::abs(phiDiff)) {
                   Mat correction;
                   CreateRotationY(-resudialPhiDiff, correction);
                   sensorDiff = sensorDiff * correction;
               }
           }

           sensorEstimate = sensorEstimate * sensorDiff;
          
           in->originalExtrinsics.copyTo(prevExtrinsics);
           prevImage = in;

           if(debug) { 
               double intPhi = GetDistanceX(sensorEstimate, Mat::eye(4, 4, CV_64F));
               double intTheta  = GetDistanceY(sensorEstimate, Mat::eye(4, 4, CV_64F));
               
               LogR << "t=" << in->id 
                  << " type=sensor-int-x"
                  << " val=" << intPhi;
               LogR << "t=" << in->id
                  << " type=sensor-int-y"
                  << " val=" << intTheta;
           }
        }
    }
};
}
#endif
