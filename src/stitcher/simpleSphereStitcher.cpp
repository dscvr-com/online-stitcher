
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
	vector<Size> predictedSizes(n);
	vector<Point> predictedCorners(n);

	for (size_t i = 0; i < n; i++) {
        auto image = in[i];
            
        Mat k;
        Mat scaledIntrinsics;
        ScaleIntrinsicsToImage(image->intrinsics, images[i], scaledIntrinsics, debug ? 10 : 1);
        From3DoubleTo3Float(scaledIntrinsics, k);
        //Decide wether to take tl/bl. We have 1 px error margin. 
        Point tl = warper.warpPoint(Point(0, image->image.rows), k, 
                cameras[i].R);

        Point bl = warper.warpPoint(Point(0, 0), k, cameras[i].R);

        Rect predictedRoi = warper.warpRoi(in[i]->image.size(), k, cameras[i].R);
        if(abs(bl.x - tl.x) > predictedRoi.width / 2) {
            //OMG - we have a corner case.
            if(bl.x < tl.x) {
                corners[i] = tl;
            } else {
                corners[i] = bl;
            }
        } else {
            //Standard case. 
            if(bl.x > tl.x) {
                corners[i] = tl;
            } else {
                corners[i] = bl;
            }
        }

        predictedSizes[i] = predictedRoi.size(); 
        predictedCorners[i] = predictedRoi.tl();
        corners[i].y = predictedCorners[i].y;


        auto candidate = imageCache.find(in[i]->id);

        // Only allow cache serving if image is there and y coordinate did not change.
        if(candidate != imageCache.end() && 
                cornerCache.at(in[i]->id).y == corners[i].y) {
            warpedImages[i] = imageCache.at(in[i]->id);
            warpedMasks[i] = maskCache.at(in[i]->id);
        } else {

            Mat mask(images[i].size(), CV_8U);
            mask.setTo(Scalar::all(255));

            //Prevent warping around. 
            if(abs(predictedCorners[i].x - corners[i].x) > 2) {
                
                //AssertWGTM(predictedSizes[i].width, 
                //    predictedSizes[i].height * 4, 
                //    "Is Corner Case");

                Mat ry4, ry3;
                CreateRotationY(M_PI, ry4);
                From4DoubleTo3Float(ry4, ry3);
                warper.warp(images[i], k, ry3 * cameras[i].R, 
                        INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
                warper.warp(mask, k, ry3 * cameras[i].R, 
                        INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
            } else {
                warper.warp(images[i], k, cameras[i].R, 
                    INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
                warper.warp(mask, k, cameras[i].R, 
                        INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);

            }
            imageCache[in[i]->id] = warpedImages[i]; 
            maskCache[in[i]->id] = warpedMasks[i]; 
        }
        cornerCache[in[i]->id] = corners[i];
        corners[i].x += cameras[i].t.at<float>(0);
        corners[i].y += cameras[i].t.at<float>(1);
    }

    //Final blending
	Mat warpedImageAsShort;
	Ptr<Blender> blender;

	blender = Blender::createDefault(Blender::FEATHER, true);
  
    Rect resultRoi = cv::detail::resultRoi(predictedCorners, predictedSizes);
    blender->prepare(resultRoi);

	for (size_t i = 0; i < n; i++)
	{
        warpedImages[i].convertTo(warpedImageAsShort, CV_16S);

        Rect imageRoi(corners[i], warpedImages[i].size()); 
        Rect overlap = imageRoi & resultRoi;
    
        Rect overlapI(0, 0, overlap.width, overlap.height);

        if(overlap.width == imageRoi.width) {
            //Image fits.
            blender->feed(warpedImageAsShort(overlapI), 
                    warpedMasks[i](overlapI), 
                    corners[i]);
        } else {
            //Image overlaps on X-Axis and we have to blend two parts.
            blender->feed(warpedImageAsShort(overlapI), 
                    warpedMasks[i](overlapI), 
                    overlap.tl());

            Rect other(resultRoi.x, overlap.y, 
                    imageRoi.width - overlap.width, overlap.height);
            Rect otherI(overlap.width, 0, other.width, other.height);
            blender->feed(warpedImageAsShort(otherI), 
                    warpedMasks[i](otherI), 
                    other.tl());
        }
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
   k.at<float>(0, 2) = 0;
   k.at<float>(1, 2) = 0;
   return warper.warpPoint(point, k, r); 
}
}
