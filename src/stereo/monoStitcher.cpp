//////////////////////////////////////////////////////////////////////////////////////
//
// Beem mono stitcher test by Emi
//
// Let there be MSV!
//
//
#define _USE_MATH_DEFINES
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

#include "../common/image.hpp"
#include "../common/assert.hpp"
#include "../math/quat.hpp"
#include "../math/support.hpp"
#include "../recorder/streamingRecorderController.hpp"
#include "../stereo/monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

struct StereoTarget {
    Mat R;
    Size size;
};

//Maps the target area from sphere to image space.
void AreaToCorners(const Size targetDimensions, const Mat &targetCenter, 
        const vector<Mat> &targetCorners, vector<Point2f> &corners) {

	Mat I = Mat::eye(4, 4, CV_64F);
    corners.resize(4);

    for(int i = 0; i < 4; i++) { 
        Mat rot = targetCenter.inv() * targetCorners[i];
        
	    corners[i].x = -tan(GetDistanceByDimension(I, rot, 0)) + 0.5;
	    corners[i].y = -tan(GetDistanceByDimension(I, rot, 1)) + 0.5;
       
        corners[i].x *= targetDimensions.width;
        corners[i].y *= targetDimensions.height;
    }
}

//Gets the outer bounding rectangle of the target area. 
//Is only consistent with the GetTargetArea method below. 
Rect CornersToRoi(const vector<Point2f> &corners) {
    float x = min2(corners[0].x, corners[3].x);
    float y = min2(corners[2].y, corners[3].y);
    float width = max2(corners[1].x, corners[2].x) - x;
    float height = max2(corners[0].y, corners[1].y) - y;

    AssertGTM(width, 0.0f, "Transformation is not mirrored.");
    AssertGTM(height, 0.0f, "Transformation is not mirrored.");
    Rect roi(x, y, width, height); 

    return roi;
}


//Extracts the target area, as points on a sphere surface, 
//encoded as rotation matrices. 
const double hBufferRatio = 3;
const double vBufferRatio = 0.05;

void GetTargetArea(const SelectionPoint &a, const SelectionPoint &b, Mat &center, vector<Mat> &corners) {
    double hLeft = a.hPos;
    double hRight = b.hPos;
    if(hLeft > hRight) {
        //Corner case - ring closing. 
        hRight += 2 * M_PI;
    }
    double hCenter = (hLeft + hRight) / 2.0;
    double vCenter = a.vPos;
    double vTop = vCenter - a.vFov / 2.0;
    double vBot = vCenter + a.vFov / 2.0;
    double hBuffer = a.hFov * hBufferRatio;
    double vBuffer = a.vFov * vBufferRatio;

    corners.resize(4);

    GeoToRot(hCenter, vCenter, center);
    GeoToRot(hLeft - hBuffer, vTop - vBuffer, corners[0]);
    GeoToRot(hRight + hBuffer, vTop - vBuffer, corners[1]);
    GeoToRot(hRight + hBuffer, vBot + vBuffer, corners[2]);
    GeoToRot(hLeft - hBuffer, vBot + vBuffer, corners[3]);

    assert(hLeft - hBuffer < hRight + hBuffer); 
}

void MapToTarget(const InputImageP a, const StereoTarget &target, Mat &result, Mat &targetK) {
    Mat rot, rot4 = target.R.inv() * a->adjustedExtrinsics;
    From4DoubleTo3Double(rot4, rot);
    double t = -0.02; //"Arm length" in homogenic space, 
    //which means 1 =  width of sensor.
    double unit[] = {t, 0.0, 0.0, 0.0};
    Mat translation = Mat::eye(3, 3, CV_64F);
    translation(Rect(2, 0, 1, 3)) = rot * Mat(3, 1, CV_64F, unit);
    translation.at<double>(2, 2) = 1;

    //cout << rot << endl;
    //cout << translation << endl;

    Mat aK;
    ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), aK);
    targetK = aK.clone();
 	targetK.at<double>(0, 2) = target.size.width / 2.0f;
 	targetK.at<double>(1, 2) = target.size.height / 2.0f;

    Mat transformation = targetK * translation * rot * aK.inv();
    Mat transformationF;
    From3DoubleTo3Float(transformation, transformationF);
    
    //cout << rot << endl;
    //cout << aK.inv() << endl;
    //cout << targetK << endl;
    //cout << transformation << endl;

    result = Mat(target.size, a->image.data.type());

    warpPerspective(a->image.data, result, transformationF, target.size, 
            INTER_LINEAR, BORDER_CONSTANT, 0);
}

void MonoStitcher::CreateStereo(const SelectionInfo &a, const SelectionInfo &b, StereoImage &stereo) {

    const static bool debug = false;

    Mat k;
    stereo.valid = false;

	AssertEQ(a.image->image.cols, b.image->image.cols);
	AssertEQ(a.image->image.rows, b.image->image.rows);

    StereoTarget target;
    vector<Mat> targetArea;
    vector<Point2f> corners;
    Mat newKA, newKB;

    GetTargetArea(a.closestPoint, b.closestPoint, target.R, targetArea);
    AreaToCorners(a.image->image.size(), target.R, targetArea, corners);
    Rect roi = CornersToRoi(corners);
    target.size = roi.size();

    Mat resA, resB;

    MapToTarget(a.image, target, resA, newKA);
    MapToTarget(b.image, target, resB, newKB);

    if(debug) {
        Point2f tl(roi.x, roi.y);
        for(size_t i = 0; i < corners.size(); i++) {
            line(resA, corners[i] - tl, 
                    corners[(i + 1) % corners.size()] - tl, 
                    Scalar(0, 0, 255), 3);

            line(resB, corners[i] - tl, 
                    corners[(i + 1) % corners.size()] - tl, 
                    Scalar(0, 0, 255), 3);
        }
        
        imwrite("dbg/" + ToString(a.image->id) + "_warped_A.jpg", resA);
        imwrite("dbg/" + ToString(a.image->id) + "_warped_B.jpg", resB);
    }

    stereo.A->image = Image(resA);
    stereo.B->image = Image(resB);

	stereo.A->intrinsics = newKA;
	stereo.A->adjustedExtrinsics = target.R;
	stereo.A->originalExtrinsics = target.R;
	stereo.A->id = a.image->id;

	stereo.B->intrinsics = newKB;
	stereo.B->adjustedExtrinsics = target.R;
	stereo.B->originalExtrinsics = target.R;
	stereo.B->id = b.image->id;

	stereo.extrinsics = target.R;

	stereo.valid = true;
}
}
