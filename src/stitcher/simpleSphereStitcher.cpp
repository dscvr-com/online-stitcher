
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "simpleSphereStitcher.hpp"
#include "../common/assert.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

StitchingResultP SimpleSphereStitcher::Stitch(const std::vector<InputImageP> &in, bool debug) {
	size_t n = in.size();
    assert(n > 0);

    vector<Mat> images(n);
    vector<cv::detail::CameraParams> cameras(n);

    for(size_t i = 0; i < n; i++) {

        auto image = in[i];

        AssertEQ(image->image.data.cols, image->image.cols);
        AssertEQ(image->image.data.rows, image->image.rows);
        AssertEQ(image->adjustedExtrinsics.cols, 4);
        AssertEQ(image->adjustedExtrinsics.rows, 4);
        AssertEQ(image->adjustedExtrinsics.type(), CV_64F);

        images[i] = image->image.data;
        From4DoubleTo3Float(image->adjustedExtrinsics, cameras[i].R);
        for(size_t j = 0; j < 3; j++)
            cameras[i].t.at<float>(j) = image->adjustedExtrinsics.at<double>(j, 3);
    }

	//Create masks and small images for fast stitching. 
    vector<Mat> masks(n);

    for(size_t i = 0; i < n; i++) {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

	//Create data structures for warped images
	vector<Point> corners(n);
	vector<Mat> warpedImages(n);
	vector<Mat> warpedMasks(n);
	vector<Size> warpedSizes(n);

	for (size_t i = 0; i < n; i++) {
        auto image = in[i];

       	Mat k;
        Mat scaledIntrinsics;
        ScaleIntrinsicsToImage(image->intrinsics, images[i], scaledIntrinsics, debug ? 10 : 1);
        From3DoubleTo3Float(scaledIntrinsics, k);

        //Big
        corners[i] = warper.warp(images[i], k, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
        corners[i].x += cameras[i].t.at<float>(0);
        corners[i].y += cameras[i].t.at<float>(1);
        warper.warp(masks[i], k, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
        warpedSizes[i] = warpedImages[i].size();
    }

    masks.clear();

    //Final blending
	Mat warpedImageAsShort;
	Ptr<Blender> blender;

	blender = Blender::createDefault(Blender::FEATHER, true);
  
    blender->prepare(corners, warpedSizes);

	for (size_t i = 0; i < n; i++)
	{
        warpedImages[i].convertTo(warpedImageAsShort, CV_16S);
		blender->feed(warpedImageAsShort, warpedMasks[i], corners[i]);
	}

	StitchingResultP res(new StitchingResult());
    Mat image;
    Mat mask;
	blender->blend(image, mask);

    res->image = Image(image);
    res->mask = Image(mask);

	//Rotate by 180°
	//flip(res->image, res->image, -1);
	//flip(res->mask, res->mask, -1);

    res->corner.x = corners[0].x;
    res->corner.y = corners[0].y;

    for(size_t i = 1; i < n; i++) {
        res->corner.x = min(res->corner.x, corners[i].x);
        res->corner.y = min(res->corner.y, corners[i].y);
    }

	return res;
}

cv::Point2f SimpleSphereStitcher::Warp(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const Size &imageSize) {
   Mat sk, r, k;
   From4DoubleTo3Float(extrinsics, r);
   ScaleIntrinsicsToImage(intrinsics, imageSize, sk);
   From3DoubleTo3Float(sk, k);
   return warper.warpPoint(Point(0, 0), k, r); 
}
}
