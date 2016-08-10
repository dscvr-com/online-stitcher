#define _USE_MATH_DEFINES
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

#include "../common/image.hpp"
#include "../common/logger.hpp"
#include "../common/assert.hpp"
#include "../math/quat.hpp"
#include "../math/support.hpp"
#include "../recorder/imageSelector.hpp"
#include "../stereo/monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

/*
 * Represents the target plane in rotational space. 
 */
struct StereoTarget {
    Mat R;
    Size size;
};

/*
 * Maps a target area given from rotational coordinates
 * to the image plane given by center.
 *
 * @param targetDimensions Size of the target image, in pixes.
 * @param targetCenter Location of the target image, rotation matrix.
 * @param targetK Intrinsic parameters of the target's camera.
 * @param targetCorners The corners of the area to map, in rotational coords. 
 */
void AreaToCorners(const Size targetDimensions, const Mat &targetCenter, 
        const Mat &targetK, const vector<Mat> &targetCorners, 
        vector<Point2f> &corners) {

	Mat I = Mat::eye(4, 4, CV_64F);
    corners.resize(4);
        
    double hFov = GetHorizontalFov(targetK);
    double vFov = GetVerticalFov(targetK); 

    for(int i = 0; i < 4; i++) { 
        Mat rot = targetCenter.inv() * targetCorners[i];
        
	    corners[i].x = -tan(GetDistanceByDimension(I, rot, 0)) / tan(hFov / 2) + 0.5;
	    corners[i].y = -tan(GetDistanceByDimension(I, rot, 1)) / tan(vFov / 2) + 0.5;
       
        corners[i].x *= targetDimensions.width / 2;
        corners[i].y *= targetDimensions.height / 2;
    }
}

/*
 * Gets the outer bounding rectangle of the target area. 
 * Is only consistent with the GetTargetArea method below. 
 */
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


const double hBufferRatio = 1.5;
const double vBufferRatio = -0.02;

/*
 * For two given images, extracts the target area, as points on a sphere surface, 
 * encoded as rotation matrices. 
 *
 * @param a The first image. 
 * @param b The second image. 
 * @param center The location of the image plane, as rotational matrix. 
 * @param corners Vector for storing the result. 
 */
void GetTargetArea(const SelectionPoint &a, const SelectionPoint &b, Mat &center, vector<Mat> &corners) {
    double hLeft = a.hPos;
    double hRight = b.hPos;

    if(a.globalId == b.globalId) {
        hLeft = a.hPos - a.hFov;
        hRight = a.hPos + a.hFov;
    }

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
    
    // TODO - this is a hard coded corner case for mono-rectify
    // which is only used for preview images. Still it is very ugly. Remove it.
    if(a.globalId == b.globalId) {
        hBuffer = a.hFov;
    }

    corners.resize(4);

    GeoToRot(hCenter, vCenter, center);
    GeoToRot(hLeft - hBuffer, vTop - vBuffer, corners[0]);
    GeoToRot(hRight + hBuffer, vTop - vBuffer, corners[1]);
    GeoToRot(hRight + hBuffer, vBot + vBuffer, corners[2]);
    GeoToRot(hLeft - hBuffer, vBot + vBuffer, corners[3]);

    Assert(hLeft - hBuffer < hRight + hBuffer);
}

/*
 * Projects a given image to it's target plane. Applies perspective 
 * correction and cuts the proper region. 
 *
 * @param a The image to project. 
 * @param target The target plane. 
 * @param result The resulting image. 
 * @param targetK The resulting intrinsics of the projected image. 
 * @param debug Debugger flag, usually set by calling method. 
 */
void MapToTarget(const InputImageP a, const StereoTarget &target, Mat &result, Mat &targetK, bool debug = false) {
    Mat rot, rot4 = target.R.inv() * a->adjustedExtrinsics;
    From4DoubleTo3Double(rot4, rot);
    double t = -0.02; //"Arm length" in homogenic space, 
    //which means 1 =  width of sensor.
    double unit[] = {t, 0.0, 0.0, 0.0};
    Mat translation = Mat::eye(3, 3, CV_64F);
    translation(Rect(2, 0, 1, 3)) = rot * Mat(3, 1, CV_64F, unit);
    translation.at<double>(1, 2) = 0;
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

    Scalar border(0);

    // If we are in debug mode, use a bright red border for undefined areas. 
    if(debug) {
        border = Scalar(0, 0, 255);
    }
    
    result = Mat(target.size, a->image.data.type(), border); 
    warpPerspective(a->image.data, result, transformationF, target.size, 
        INTER_LINEAR, BORDER_CONSTANT, border); 

    if(debug) { 
        vector<Point2f> corners = {
            Point2f(0, 0),
            Point2f(0, target.size.height),
            Point2f(target.size.width, target.size.height),
            Point2f(target.size.width, 0)
        };

        perspectiveTransform(corners, corners, transformationF.inv());

        for(size_t i = 0; i < corners.size(); i++) {
            line(a->image.data, corners[i], 
                    corners[(i + 1) % corners.size()], 
                    Scalar(255, 0, 0), 3);
        }
    }
}

InputImageP MonoStitcher::RectifySingle(const SelectionInfo &a) {
    StereoTarget target;
    vector<Mat> targetArea;
    vector<Point2f> corners;

    // Set our projection target to the selection point of the image. 
    target.R = a.closestPoint.extrinsics;

    // Calculate target area. 
    GetTargetArea(a.closestPoint, a.closestPoint, target.R, targetArea);
    AreaToCorners(a.image->image.size(), target.R, a.image->intrinsics, 
            targetArea, corners);

    // Create target image and set the size of the projectiont target. 
    Rect roi = CornersToRoi(corners);
    target.size = roi.size();

    // Map our image to our predefined target. 
    Mat resA, newKA;
    MapToTarget(a.image, target, resA, newKA);

    // Construct resulting image. 
    InputImageP res = std::make_shared<InputImage>(); 

    res->image = Image(resA);
    res->intrinsics = newKA;
    res->originalExtrinsics = target.R.clone(); 
    res->adjustedExtrinsics = target.R.clone(); 
    res->id = a.image->id;

    return res;
}

void MonoStitcher::CreateStereo(const SelectionInfo &a, const SelectionInfo &b, StereoImage &stereo) {

    const static bool debug = false;
    AssertFalseInProduction(debug);

    Mat k;
    stereo.valid = false;

        //AssertEQ(a.image->image.cols, b.image->image.cols);
        //AssertEQ(a.image->image.rows, b.image->image.rows);
    AssertMatEQ<double>(a.image->intrinsics, b.image->intrinsics);

    StereoTarget target;
    vector<Mat> targetArea;
    vector<Point2f> corners;
    Mat newKA, newKB;

    // Get the target area which lies between the two given selection points. 
    GetTargetArea(a.closestPoint, b.closestPoint, target.R, targetArea);

    // Calculate target size on projection plane. 
    AreaToCorners(a.image->image.size(), target.R, a.image->intrinsics, 
            targetArea, corners);

    Rect roi = CornersToRoi(corners);
    target.size = roi.size();

    Mat resA, resB;

    // Map both images to the same projection target. 
    MapToTarget(a.image, target, resA, newKA, debug);
    MapToTarget(b.image, target, resB, newKB, debug);

    // If debug is on, draw all important regions and save the images. 
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
        
        imwrite("dbg/" + ToString(a.image->id) + "_input_A.jpg", a.image->image.data);
        imwrite("dbg/" + ToString(a.image->id) + "_input_B.jpg", b.image->image.data);
        imwrite("dbg/" + ToString(a.image->id) + "_warped_A.jpg", resA);
        imwrite("dbg/" + ToString(a.image->id) + "_warped_B.jpg", resB);
    }

    // Construct resulting stereo image structure. 
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
