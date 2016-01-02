#include <iostream>
#include <algorithm>

#include "../stitcher/simpleSphereStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

void CreateRotationPhiTheta(const double phi, const double theta, Mat &res) {
    Mat rPhi, rTheta;

    CreateRotationY(phi, rPhi);
    CreateRotationX(theta, rTheta);

    res = rPhi * rTheta;
}

int main(int, char**) {

    const bool drawDebug = false;

    SimpleSphereStitcher debugger;

    Mat start;
    Mat end;

    // For equality testing, only 1d lerp supported. 
    // Veryfiy others by drawDebug = true. 
    double phiStart = 0, phiEnd = M_PI / 2;
    double thetaStart = 0, thetaEnd = 0;

    CreateRotationPhiTheta(phiStart, thetaStart, start);
    CreateRotationPhiTheta(phiEnd, thetaEnd, end);

    Mat intrinsics = Mat::eye(3, 3, CV_64F);
    intrinsics.at<double>(0, 2) = 2;
    intrinsics.at<double>(1, 2) = 2;

    Mat canvas = Mat::zeros(2000, 2000, CV_8UC3);

    for(double i = 0; i <= 1; i += 0.1) {
       Mat slerp = Mat::eye(4, 4, CV_64F);
       Mat lerp;

       CreateRotationPhiTheta(phiStart * (1 - i) + phiEnd * i, 
               thetaStart * (1 - i) + thetaEnd * i, lerp);
       
       Slerp(start, end, i, slerp); 

       Point projectedSlerp = debugger.WarpPoint(intrinsics, slerp, 
               Size(2, 2), Point(1, 1));
       Point projectedLerp = debugger.WarpPoint(intrinsics, lerp, 
               Size(2, 2), Point(1, 1));

       if(drawDebug) {
           cv::circle(canvas, projectedLerp, 8, 
                Scalar(0x00, 0xFF, 0x00), -1);
           
           cv::circle(canvas, projectedSlerp, 8, 
                Scalar(0x00, 0x00, 0xFF), -1);

           if(!drawDebug) {
                AssertEQ(projectedSlerp, projectedLerp);
           }
       }
    }

    if(drawDebug) {
        imwrite("dbg/slerp.png", canvas);
    }
    
    cout << "[\u2713] Slerp support module." << endl;

    return 0;
}
