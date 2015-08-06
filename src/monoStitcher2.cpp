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
	result->valid = false;

	int width = a->img.cols / 10;
	int height = a->img.rows;
	int offset = width * 4;

	Rect leftRoi(offset - width, 0, width, height);
	Rect rightRoi(a->img.cols - offset - width, 0, width, height);

 	result->A.img = a->img(leftRoi);
	result->B.img = a->img(rightRoi);
	//result->A.img = a->img;
	//result->B.img = a->img;

	assert(result->A.img.rows == height && result->A.img.cols == width);
	assert(result->B.img.rows == height && result->B.img.cols == width);

	result->A.intrinsics = a->intrinsics.clone();
	result->A.extrinsics = a->extrinsics.clone();
	cout << "A intr: " << result->A.intrinsics << endl;
	result->A.intrinsics.at<double>(0, 2) *= width / (double)a->img.cols;
	result->A.id = a->id;

	result->B.intrinsics = a->intrinsics.clone();
	result->B.extrinsics = a->extrinsics.clone();
	
	assert(result->B.extrinsics.type() == CV_64F);
	assert(result->B.intrinsics.type() == CV_64F);

	result->B.intrinsics.at<double>(0, 2) *= width / (double)a->img.cols;
	result->B.id = a->id;

	result->extrinsics = a->extrinsics.clone();

	result->valid = true;
	return result;
}

}