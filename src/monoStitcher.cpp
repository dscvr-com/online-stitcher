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
StereoImageP MonoStitcher::CreateStereo(ImageP a, ImageP b, StereoTarget target) {
	Mat k;
    
    //Avoid unused param warning. 
    assert(target.center.cols == 4);

	StereoImageP result(new StereoImage());
	result->valid = false;

	assert(a->img.cols == b->img.cols);
	assert(a->img.rows == b->img.rows);

	//cout << "AR: " << a->extrinsics << endl;
	//cout << "BR: " << b->extrinsics << endl;

	Mat rot = a->extrinsics.inv() * b->extrinsics;
    
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

	ScaleIntrinsicsToImage(a->intrinsics, a->img, aIntrinsics); 
	ScaleIntrinsicsToImage(b->intrinsics, b->img, bIntrinsics);
    
	Mat transA = aIntrinsics * rotN.inv() * aIntrinsics.inv();
	Mat transB = bIntrinsics * rotN * bIntrinsics.inv();

	Mat resA(a->img.rows, a->img.cols, CV_32F);
	Mat resB(b->img.rows, b->img.cols, CV_32F);

    Mat ai4;
    From3DoubleTo4Double(aIntrinsics, ai4);

	Mat I = Mat::eye(4, 4, CV_64F);
	vector<Point2f> corners(4);
        
    double maxHFov = GetHorizontalFov(aIntrinsics);
    double maxVFov = GetVerticalFov(aIntrinsics); 
    
    for(int i = 0; i < 4; i++) { 
        
        //Mat rot = target.center.inv() * target.corners[i] * a->offset; 
        Mat rot = target.center.inv() * target.corners[i]; 
        Mat rot4;
        From3DoubleTo4Double(rot, rot4);
        
	    corners[i].x = -tan(GetDistanceByDimension(I, rot4, 0)) / tan(maxHFov) + 0.5;
	    corners[i].y = -tan(GetDistanceByDimension(I, rot4, 1)) / tan(maxVFov) + 0.5;
        cout << "Corners A " << i << corners[i] << endl;
        corners[i].x *= a->img.cols;
        corners[i].y *= a->img.rows;
        cout << "Corners B " << i << corners[i] << endl;
        //cout << "Corner " << i << corners[i] << endl;
        //cout << "MatDiff: " << rot << endl;
    }
	//transB.at<float>(0, 2) = -tan(GetDistanceByDimension(I, rot, 0)) * a->img.cols;
	//transB.at<float>(1, 2) = -tan(GetDistanceByDimension(I, rot, 1)) * a->img.rows;

	warpPerspective(a->img, resA, transA, resA.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
	warpPerspective(b->img, resB, transB, resB.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
    
    for(size_t i = 0; i < corners.size(); i++) {
        line(resA, corners[i], corners[(i + 1) % corners.size()], Scalar(0, 0, 255), 3);
        line(resB, corners[i], corners[(i + 1) % corners.size()], Scalar(0, 0, 255), 3);
    } 

	imwrite("dbg/warped_" + ToString(a->id) + "A.jpg", resA);
	imwrite("dbg/warped_" + ToString(a->id) + "B.jpg", resB);

	Mat rvec(4, 1, CV_64F);
	ExtractRotationVector(rot, rvec);

    float x = min2(corners[0].x, corners[3].x);
    float y = min2(corners[2].y, corners[3].y);
    float width = max2(corners[1].x, corners[2].x) - x;
    float height = max2(corners[0].y, corners[1].y) - y;
    Rect roi(x, y, width, height); 

    result->A->img = resA(roi);
    result->B->img = resB(roi);

	//cout << "Diff for " << a->id << " " << rvec.t() << endl;

	Mat newKA(3, 3, CV_64F);
 	newKA.at<double>(0, 0) = aIntrinsics.at<double>(0, 0);
 	newKA.at<double>(1, 1) = aIntrinsics.at<double>(1, 1);
 	newKA.at<double>(0, 2) = width / 2.0f;
 	newKA.at<double>(1, 2) = height / 2.0f;

	result->A->intrinsics = newKA;
	//result->A->extrinsics = target.center;
	result->A->extrinsics = a->extrinsics * rotN4;
	result->A->id = a->id;

	Mat newKB = Mat::eye(3, 3, CV_64F);
 	newKB.at<double>(0, 0) = bIntrinsics.at<double>(0, 0);
 	newKB.at<double>(1, 1) = bIntrinsics.at<double>(1, 1);
 	newKB.at<double>(0, 2) = width / 2.0f;
 	newKB.at<double>(1, 2) = height / 2.0f;

	result->B->intrinsics = newKB;
	result->B->extrinsics = b->extrinsics * rotN4.inv();
	result->B->id = b->id;

	result->extrinsics = rotN4.inv() * b->extrinsics;

	result->valid = true;
	return result;
}
}
