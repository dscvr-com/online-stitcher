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

void GetCorners(vector<Point2f> &corners, const StereoTarget &target, const Mat &intrinsics, int width, int height) {
    
    double maxHFov = GetHorizontalFov(intrinsics);
    double maxVFov = GetVerticalFov(intrinsics); 
	Mat I = Mat::eye(4, 4, CV_64F);

    for(int i = 0; i < 4; i++) { 
       
        //Todo: Don't we need some offset here?  

        
        Mat rot = target.center.inv() * target.corners[i];
        
	    corners[i].x = -tan(GetDistanceByDimension(I, rot, 0)) / tan(maxHFov) + 0.5;
	    corners[i].y = -tan(GetDistanceByDimension(I, rot, 1)) / tan(maxVFov) + 0.5;
       
        //cout << "Corners A " << i << corners[i] << endl;
        corners[i].x *= width;
        corners[i].y *= height;
        //cout << "Corners B " << i << corners[i] << endl;
        //cout << "Corner " << i << corners[i] << endl;
        //cout << "MatDiff: " << rot << endl;
    }
}

StereoImageP MonoStitcher::CreateStereo(SelectionInfo a, SelectionInfo b, StereoTarget target) {
	Mat k;
    
    //Avoid unused param warning. 
    assert(target.center.cols == 4);

	StereoImageP result(new StereoImage());
	result->valid = false;

	assert(a.image->img.cols == b.image->img.cols);
	assert(a.image->img.rows == b.image->img.rows);

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

	ScaleIntrinsicsToImage(a.image->intrinsics, a.image->img, aIntrinsics); 
	ScaleIntrinsicsToImage(b.image->intrinsics, b.image->img, bIntrinsics);

    Mat aOffset;
    From4DoubleTo3Double(a.image->extrinsics.inv() * a.closestPoint.extrinsics, aOffset);

    Mat bOffset; 
    From4DoubleTo3Double(b.image->extrinsics.inv() * b.closestPoint.extrinsics, bOffset);
    
	Mat transA = aIntrinsics * aOffset.inv() * rotN.inv() * aIntrinsics.inv();
	Mat transB = bIntrinsics * bOffset.inv() * rotN * bIntrinsics.inv();

	Mat resA(a.image->img.rows, a.image->img.cols, CV_32F);
	Mat resB(b.image->img.rows, b.image->img.cols, CV_32F);

	Mat I = Mat::eye(4, 4, CV_64F);
	vector<Point2f> cornersA(4);
	vector<Point2f> cornersB(4);
       
    GetCorners(cornersA, target, aIntrinsics, a.image->img.cols, a.image->img.rows);
    GetCorners(cornersB, target, bIntrinsics, b.image->img.cols, a.image->img.rows);

	warpPerspective(a.image->img, resA, transA, resA.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
	warpPerspective(b.image->img, resB, transB, resB.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    
    /*for(size_t i = 0; i < cornersA.size(); i++) {
        line(resA, cornersA[i], cornersA[(i + 1) % cornersA.size()], Scalar(0, 0, 255), 3);
        line(resB, cornersB[i], cornersB[(i + 1) % cornersB.size()], Scalar(0, 0, 255), 3);
    }

	imwrite("dbg/warped_" + ToString(a->id) + "A.jpg", resA);
	imwrite("dbg/warped_" + ToString(a->id) + "B.jpg", resB);
*/
	Mat rvec(4, 1, CV_64F);
	ExtractRotationVector(rot, rvec);

    result->A->img = resA(CornersToRoi(cornersA));
    result->B->img = resB(CornersToRoi(cornersB));

	//cout << "Diff for " << a->id << " " << rvec.t() << endl;

	Mat newKA(3, 3, CV_64F);
 	newKA.at<double>(0, 0) = aIntrinsics.at<double>(0, 0);
 	newKA.at<double>(1, 1) = aIntrinsics.at<double>(1, 1);
 	newKA.at<double>(0, 2) = result->A->img.cols / 2.0f;
 	newKA.at<double>(1, 2) = result->B->img.rows / 2.0f;

	result->A->intrinsics = newKA;
	//result->A->extrinsics = target.center;
	result->A->extrinsics = target.center;
	result->A->id = a.image->id;

    //Todo: Focal len not correct. 

	Mat newKB = Mat::eye(3, 3, CV_64F);
 	newKB.at<double>(0, 0) = bIntrinsics.at<double>(0, 0);
 	newKB.at<double>(1, 1) = bIntrinsics.at<double>(1, 1);
 	newKB.at<double>(0, 2) = result->B->img.cols / 2.0f;
 	newKB.at<double>(1, 2) = result->B->img.rows / 2.0f;

	result->B->intrinsics = newKB;
	//result->B->extrinsics = target.center;
	result->B->extrinsics = target.center;
	result->B->id = a.image->id + 100000;

	result->extrinsics = rotN4.inv() * b.image->extrinsics;

	result->valid = true;
	return result;
}
}
