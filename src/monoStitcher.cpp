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

#include "core.hpp"
#include "quat.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

StereoImage *CreateStereo(Image *a, Image *b) {
	Mat k;

	StereoImage* result = new StereoImage();

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

	//Todo - don't we need a lerping here? 
	quat::Mult(rotQ, 0.5f, avg);

	Mat rotN = Mat::eye(3, 3, CV_64F);
	quat::ToMat(avg, rotN);


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

	vector<Point2f> cornersA(4);
	cornersA[0] = cvPoint(0, 0);
	cornersA[1] = cvPoint(a->img.cols, 0);
	cornersA[2] = cvPoint(a->img.cols, a->img.rows);
	cornersA[3] = cvPoint(0, a->img.rows);
	vector<Point2f> cornersB(4);
	cornersB[0] = cvPoint(0, 0);
	cornersB[1] = cvPoint(b->img.cols, 0);
	cornersB[2] = cvPoint(b->img.cols, b->img.rows);
	cornersB[3] = cvPoint(0, b->img.rows);

	Mat I = Mat::eye(4, 4, CV_64F);
	transB.at<float>(0, 2) = -GetDistanceByDimension(I, rot, 0);
	transB.at<float>(1, 2) = -GetDistanceByDimension(I, rot, 1);

	perspectiveTransform(cornersA, cornersA, transA);
	perspectiveTransform(cornersB, cornersB, transB);

	warpPerspective(a->img, resA, transA, resA.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
	warpPerspective(b->img, resB, transB, resB.size(), INTER_LINEAR, BORDER_CONSTANT, 0);

	int x = max2(cornersB[0].x, cornersB[3].x);
	x = max2(x, 0);
	int width = min2(cornersA[1].x, cornersA[2].x) - x;
	width = max(0, width);

	int y = max4(interpolate(x, cornersB[0].x, cornersB[1].x, cornersB[0].y, cornersB[1].y), 
		interpolate(x, cornersA[0].x, cornersA[1].x, cornersA[0].y, cornersA[1].y),
		interpolate(x + width, cornersB[0].x, cornersB[1].x, cornersB[0].y, cornersB[1].y), 
		interpolate(x + width, cornersA[0].x, cornersA[1].x, cornersA[0].y, cornersA[1].y));
	y = max2(y, 0);

	int height = min4(interpolate(x, cornersB[3].x, cornersB[2].x, cornersB[3].y, cornersB[2].y), 
		interpolate(x, cornersA[3].x, cornersA[2].x, cornersA[3].y, cornersA[2].y),
		interpolate(x + width, cornersB[3].x, cornersB[2].x, cornersB[3].y, cornersB[2].y), 
		interpolate(x + width, cornersA[3].x, cornersA[2].x, cornersA[3].y, cornersA[2].y)) - y;

	height = max(0, height);
	Rect finalRoi(x, y, width, height);

	try {
	 	result->A.img = resA(finalRoi);
		result->B.img = resB(finalRoi);
	} catch(Exception e) {
		cout << "ROI error - an image might be missing. Skipping." << endl;
		return result;
	}

	if(finalRoi.width <= 0 || finalRoi.height <= 0) {
		cout << "Zero ROI - Skipping" << endl;
		return result; 
	}

	Mat newKA(3, 3, CV_64F);
 	newKA.at<double>(0, 0) = aIntrinsics.at<double>(0, 0);
 	newKA.at<double>(1, 1) = aIntrinsics.at<double>(1, 1);
 	newKA.at<double>(0, 2) = width / 2.0f;
 	newKA.at<double>(1, 2) = height / 2.0f;

 	Mat rotN4;
 	From3DoubleTo4Double(rotN, rotN4);

	result->A.intrinsics = newKA;
	result->A.extrinsics = a->extrinsics * rotN4;
	result->A.id = a->id;

	Mat newKB = Mat::eye(3, 3, CV_64F);
 	newKB.at<double>(0, 0) = bIntrinsics.at<double>(0, 0);
 	newKB.at<double>(1, 1) = bIntrinsics.at<double>(1, 1);
 	newKB.at<double>(0, 2) = width / 2.0f;
 	newKB.at<double>(1, 2) = height / 2.0f;

	result->B.intrinsics = newKB;
	result->B.extrinsics = a->extrinsics * rotN4;
	result->B.id = b->id;

	result->extrinsics = a->extrinsics * rotN4;

	result->valid = true;
	return result;
}

}