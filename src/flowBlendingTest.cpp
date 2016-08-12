#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/tracking.hpp>

#include "minimal/imagePreperation.hpp"
#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "imgproc/planarCorrelator.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

cv::Vec3b Sample(const cv::Mat& img, const float inx, const float iny)
{
    assert(!img.empty());
    assert(img.channels() == 3);

    int x = (int)inx;
    int y = (int)iny;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

    float a = inx - (float)x;
    float c = iny - (float)y;

    uchar b = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[0] * a) * (1.f - c)
    + (img.at<cv::Vec3b>(y1, x0)[0] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[0] * a) * c);
    uchar g = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[1] * a) * (1.f - c)
    + (img.at<cv::Vec3b>(y1, x0)[1] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[1] * a) * c);
    uchar r = (uchar)cvRound((img.at<cv::Vec3b>(y0, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y0, x1)[2] * a) * (1.f - c)
    + (img.at<cv::Vec3b>(y1, x0)[2] * (1.f - a) + img.at<cv::Vec3b>(y1, x1)[2] * a) * c);

    return cv::Vec3b(b, g, r);
}

typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> AlignerToUse;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    Assert(n % 2 == 0);

    SimpleSphereStitcher stitcher;

    for(size_t i = 0; i < n - 1; i += 1) {
        auto imgA = images[i];
        auto imgB = images[i + 1];

        /*
        Mat rotY, rotX;
        CreateRotationY(result.angularOffset.y, rotY);
        CreateRotationX(-result.angularOffset.x, rotX);
        imgA->adjustedExtrinsics = imgA->adjustedExtrinsics * rotY * rotX; 
        imgA->originalExtrinsics = imgA->adjustedExtrinsics;
        */
        
        //auto scene = stitcher.Stitch({imgA, imgB});
        //imwrite("dbg/" + ToString(i) + "_scene_multi_band_blend.jpg", scene->image.data);

        Mat overlapA, overlapB; 
        Point q;


        GetOverlappingRegionWarper(imgA, imgB, imgA->image, imgB->image, 
                overlapA, overlapB, 0, q);
        
        Mat corr; //Debug image used to print the correlation result.  
        PlanarCorrelationResult result = AlignerToUse::Align(overlapA, overlapB, corr, 0.25, 0.01, 0);

       // if(!result.valid) {
       //     cout << "Correlation: Rejected " << result.rejectionReason << "." << endl;
       //     continue;
       // }
       //
        cout << "Corr: " << result.offset << endl;

        Rect roiA(result.offset.x / -2, result.offset.y / -2, overlapA.cols, overlapA.rows);
        Rect roiB(result.offset.x / 2, result.offset.y / 2, overlapB.cols, overlapB.rows);

        Rect overlappingArea = roiA & roiB;

        Rect overlapAreaA(overlappingArea.tl() + roiA.tl(), overlappingArea.size()); 
        Rect overlapAreaB(overlappingArea.tl() + roiB.tl(), overlappingArea.size());        

        Mat grayA, grayB;

        cvtColor(overlapA(overlapAreaA), grayA, COLOR_BGR2GRAY);
        cvtColor(overlapB(overlapAreaB), grayB, COLOR_BGR2GRAY);

        //imwrite("dbg/" + ToString(i) + "_overlap_a.jpg", overlapA(overlapAreaA));
        //imwrite("dbg/" + ToString(i) + "_overlap_b.jpg", overlapB(overlapAreaB));

        Mat flow(overlapA.size(), CV_32FC2, Scalar::all(0.f));
        Mat cutFlow = flow(overlapAreaB);

        cout << "Farneback go... ";
        calcOpticalFlowFarneback(grayA, grayB, cutFlow, 0.5, 3, 4, 3, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
        cout << "done." << endl;

        for (int y = 0; y < overlapA.rows; ++y)
        {
            for (int x = 0; x < overlapA.cols; ++x)
            {
                auto d = flow.at<Vec2f>(y, x);
                flow.at<Vec2f>(y, x) = Vec2f(d(0) + result.offset.x, d(1) + result.offset.y);
            }
        }
    
        Mat flowViz(flow.size(), CV_8UC3, Scalar::all(0));

        for(int k = 0; k < flow.cols; k++) {
            for(int j = 0; j < flow.rows; j++) {
                auto f = flow.at<Vec2f>(j, k);
                flowViz.at<Vec3b>(j, k) = 
                    Vec3b(
                            std::min(255, (int)std::abs(f(0)) * 10), 
                            std::min(255, (int)std::abs(f(1)) * 10), 
                            0);
            }
        }

        //imwrite("dbg/" + ToString(i) + "_flow_farneback.jpg", flowViz);

        Mat blendedOverlap = Mat(overlapA.size(), CV_8UC3, Scalar::all(0));
        Mat mask(overlapA.size(), CV_8U, Scalar::all(255));
        mask(Rect(mask.cols - 2, 0, 1, mask.rows)).setTo(Scalar(0));

        Mat weight;
        createWeightMap(mask, 0.003, weight);
        
        //imwrite("dbg/" + ToString(i) + "_weight_map.jpg", (Mat)(weight * 255));

        Assert(overlapA.size() == overlapB.size());
       
        for (int y = 0; y < overlapA.rows; ++y)
        {
            for (int x = 0; x < overlapA.cols; ++x)
            {
                float w = weight.at<float>(y, x); //weight_row[x];
                float wm = 1 - w;
                Vec2f d = flow.at<Vec2f>(y, x);
                blendedOverlap.at<Vec3b>(y, x) = 
                    Sample(overlapA, x - d(0) * wm, y - d(1) * wm) * w + 
                    Sample(overlapB, x + d(0) * w, y + d(1) * w) * wm;
            }
        }

        imwrite("dbg/" + ToString(i) + "_scene_flow_blend.bmp", blendedOverlap);
    }
    return 0;
}

