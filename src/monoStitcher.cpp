////////////////////////////////////////////////////////////////////////////////////////
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

#include "image.hpp"
#include "quat.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "recorderController.hpp"
#include "monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

Rect CornersToRoi(const vector<Point2f> &corners) {
    float x = min2(corners[0].x, corners[3].x);
    float y = min2(corners[2].y, corners[3].y);
    float width = max2(corners[1].x, corners[2].x) - x;
    float height = max2(corners[0].y, corners[1].y) - y;
    Rect roi(x, y, width, height); 

    return roi;
}

void GetCorners(vector<Point2f> &corners, const SelectionEdge &target, const Mat, int width, int height) {
    
	Mat I = Mat::eye(4, 4, CV_64F);

    for(int i = 0; i < 4; i++) { 
       
        //Todo: Don't we need some offset here?  

        Mat rot = target.roiCenter.inv() * target.roiCorners[i];
        
        //cout << "Rot " << i << " " << rot << endl;
        
	    corners[i].x = -tan(GetDistanceByDimension(I, rot, 0)) + 0.5;
	    corners[i].y = -tan(GetDistanceByDimension(I, rot, 1)) + 0.5;
       
        corners[i].x *= width;
        corners[i].y *= height;
        //cout << "Corner " << i << corners[i] << endl;
        //cout << "MatDiff: " << rot << endl;
    }
}

void MonoStitcher::CreateStereo(const SelectionInfo &a, const SelectionInfo &b, const SelectionEdge &target, StereoImage &stereo) {

    const static bool debug = false;

    Mat k;
    stereo.valid = false;

	assert(a.image->image.cols == b.image->image.cols);
	assert(a.image->image.rows == b.image->image.rows);

	//cout << "AR: " << a->extrinsics << endl;
	//cout << "BR: " << b->extrinsics << endl;

	Mat rot = a.closestPoint.extrinsics.inv() * b.closestPoint.extrinsics;
    
	//cout << "R: " << rot << endl;

	Mat rotQ(4, 1, CV_64F);
	Mat avg(4, 1, CV_64F);
	quat::FromMat(rot, rotQ);
	
	//cout << "AQ: " << rotQ << endl;

	//Todo - write unit test for quat functions. 
	quat::Mult(rotQ, 0.5f, avg);

	Mat rotN = Mat::eye(3, 3, CV_64F);
	quat::ToMat(avg, rotN);

 	Mat rotN4;
 	From3DoubleTo4Double(rotN, rotN4);

	//cout << "rotN: " << rotN << endl;
	//cout << "rotNI: " << rotN.inv() << endl;

	Mat aIntrinsics;
	Mat bIntrinsics;

	ScaleIntrinsicsToImage(a.image->intrinsics, a.image->image, aIntrinsics); 
	ScaleIntrinsicsToImage(b.image->intrinsics, b.image->image, bIntrinsics);

    Mat aOffset;
    From4DoubleTo3Double(a.image->adjustedExtrinsics.inv() * a.closestPoint.extrinsics, aOffset);

    Mat bOffset; 
    From4DoubleTo3Double(b.image->adjustedExtrinsics.inv() * b.closestPoint.extrinsics, bOffset);
    
	Mat transA = aIntrinsics * aOffset.inv() * rotN.inv() * aIntrinsics.inv();
	Mat transB = bIntrinsics * bOffset.inv() * rotN * bIntrinsics.inv();

	Mat resA(a.image->image.rows, a.image->image.cols, CV_32F);
	Mat resB(b.image->image.rows, b.image->image.cols, CV_32F);

	Mat I = Mat::eye(4, 4, CV_64F);
	vector<Point2f> cornersA(4);
	vector<Point2f> cornersB(4);
       
    GetCorners(cornersA, target, aIntrinsics, a.image->image.cols, a.image->image.rows);
    GetCorners(cornersB, target, bIntrinsics, b.image->image.cols, a.image->image.rows);

    assert(a.image->image.IsLoaded());
    assert(b.image->image.IsLoaded());

	warpPerspective(a.image->image.data, resA, transA, resA.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
	warpPerspective(b.image->image.data, resB, transB, resB.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    
    if(debug) {
        for(size_t i = 0; i < cornersA.size(); i++) {
            line(resA, cornersA[i], cornersA[(i + 1) % cornersA.size()], Scalar(0, 0, 255), 3);
            line(resB, cornersB[i], cornersB[(i + 1) % cornersB.size()], Scalar(0, 0, 255), 3);
        }

        imwrite("dbg/warped_" + ToString(a.image->id) + "A.jpg", resA);
        imwrite("dbg/warped_" + ToString(a.image->id) + "B.jpg", resB);
    }
	Mat rvec(4, 1, CV_64F);
	ExtractRotationVector(rot, rvec);

    stereo.A->image = Image(resA(CornersToRoi(cornersA)));
    stereo.B->image = Image(resB(CornersToRoi(cornersB)));

	Mat newKA = Mat::eye(3, 3, CV_64F);
 	newKA.at<double>(0, 0) = aIntrinsics.at<double>(0, 0);
 	newKA.at<double>(1, 1) = aIntrinsics.at<double>(1, 1);
 	newKA.at<double>(0, 2) = stereo.A->image.cols / 2.0f;
 	newKA.at<double>(1, 2) = stereo.B->image.rows / 2.0f;

	stereo.A->intrinsics = newKA;
	stereo.A->adjustedExtrinsics = target.roiCenter;
	stereo.A->originalExtrinsics = target.roiCenter;
	stereo.A->id = a.image->id;

	Mat newKB = Mat::eye(3, 3, CV_64F);
 	newKB.at<double>(0, 0) = bIntrinsics.at<double>(0, 0);
 	newKB.at<double>(1, 1) = bIntrinsics.at<double>(1, 1);
 	newKB.at<double>(0, 2) = stereo.B->image.cols / 2.0f;
 	newKB.at<double>(1, 2) = stereo.B->image.rows / 2.0f;

	stereo.B->intrinsics = newKB;
	stereo.B->adjustedExtrinsics = target.roiCenter;
	stereo.B->originalExtrinsics = target.roiCenter;
	stereo.B->id = b.image->id;

	stereo.extrinsics = target.roiCenter;

	stereo.valid = true;
}
}
