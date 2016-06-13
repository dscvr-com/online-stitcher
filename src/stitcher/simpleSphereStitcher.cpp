
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

StitchingResultP SimpleSphereStitcher::Stitch(const std::vector<InputImageP> &in, bool smallImages, bool drawRotationCenters) {
    //TODO: Cleanup this mess. 
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

	//Create data structures for warped images
	vector<Point> corners(n);
	vector<Mat> warpedImages(n);
	vector<Mat> warpedMasks(n);
    vector<Size> sizes(n);

	for (size_t i = 0; i < n; i++) {
        auto image = in[i];
            
        Mat k;
        Mat scaledIntrinsics;
        ScaleIntrinsicsToImage(image->intrinsics, images[i], scaledIntrinsics, 
                smallImages ? 10 : 1);
        From3DoubleTo3Float(scaledIntrinsics, k);

        Mat mask(images[i].size(), CV_8U);
        mask.setTo(Scalar::all(255));

        corners[i] = warper.warp(images[i], k, cameras[i].R, 
            INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
        sizes[i] = warpedImages[i].size();
        warper.warp(mask, k, cameras[i].R, 
                INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);

        corners[i].x += cameras[i].t.at<float>(0);
        corners[i].y += cameras[i].t.at<float>(1);
    }

    //Final blending
	Mat warpedImageAsShort;
	Ptr<Blender> blender;

	blender = Blender::createDefault(Blender::FEATHER, true);
  
    Rect resultRoi = cv::detail::resultRoi(corners, sizes);
    blender->prepare(resultRoi);

	for (size_t i = 0; i < n; i++)
	{
        warpedImages[i].convertTo(warpedImageAsShort, CV_16S);

        blender->feed(warpedImageAsShort, 
                warpedMasks[i], 
                corners[i]);
	}

	StitchingResultP res(new StitchingResult());
    Mat image;
    Mat mask;
	blender->blend(image, mask);
   
    if(drawRotationCenters) { 
        for(size_t i = 0; i < n; i++) {    
            Mat k;
            Mat scaledIntrinsics;
            auto img = in[i];

            ScaleIntrinsicsToImage(img->intrinsics, img->image.size(), 
                    scaledIntrinsics, smallImages ? 10 : 1);

            From3DoubleTo3Float(scaledIntrinsics, k);

            Point c = warper.warpPoint(
                    Point(img->image.cols / 2, img->image.rows / 2),
                    k, cameras[i].R);

            c = c - resultRoi.tl();

            // Draw smth. 
            cv::circle(image, c, 3, Scalar(0, 0, 255), -1);
        }
    }

    res->image = Image(image);
    res->mask = Image(mask);

	//Rotate by 180°
	//flip(res->image, res->image, -1);
	//flip(res->mask, res->mask, -1);

    res->corner.x = resultRoi.x;
    res->corner.y = resultRoi.y;

	return res;
}

cv::Rect SimpleSphereStitcher::Warp(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const Size &imageSize) {
   Mat sk, r, k;
   From4DoubleTo3Float(extrinsics, r);
   ScaleIntrinsicsToImage(intrinsics, imageSize, sk);
   From3DoubleTo3Float(sk, k);
   return warper.warpRoi(imageSize, k, r); 
}
        
cv::Point SimpleSphereStitcher::WarpPoint(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const Size &imageSize, const Point &point) {
   Mat sk, r, k;
   From4DoubleTo3Float(extrinsics, r);
   ScaleIntrinsicsToImage(intrinsics, imageSize, sk);
   From3DoubleTo3Float(sk, k);
   //k.at<float>(0, 2) = 0;
   //k.at<float>(1, 2) = 0;
   return warper.warpPoint(point + Point(imageSize.width / 2, imageSize.height / 2), k, r); 
}
}
