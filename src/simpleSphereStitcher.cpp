
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "core.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "support.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
vector<Image*> RStitcher::PrepareMatrices(vector<Image*> r) {
    

    //Orient around first image (Correct orientation from start.)
    Mat center = r[0]->intrinsics.inv();
    vector<Mat> matrices(r.size());

    for(size_t i = 0; i <  r.size(); i++) {
        matrices[i] = center * r[i]->intrinsics;
    }

    //Do wave correction
    waveCorrect(matrices, WAVE_CORRECT_HORIZ);

    for(size_t i = 0; i <  r.size(); i++) {
        r[i]->intrinsics = matrices[i]; 
    }

    return r;
}

StitchingResult *RStitcher::Stitch(std::vector<Image*> in) {
	size_t n = in.size();
    assert(n > 0);

    vector<Mat> images(n);
    vector<cv::detail::CameraParams> cameras(n);

    for(size_t i = 0; i < n; i++) {
        images[i] = in[i]->img;
        cameras[i].R = in[i]->extrinsics;
    }

	//Create masks and small images for fast stitching. 
    vector<Mat> masks(n);

    vector<Mat> miniImages(n);
    vector<Mat> miniMasks(n);

    for(size_t i = 0; i < n; i++) {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));

        resize(images[i], miniImages[i], Size(), workScale, workScale, INTER_NEAREST);

        miniMasks[i].create(miniImages[i].size(), CV_8U);
        miniMasks[i].setTo(Scalar::all(255));
    }

    //Create warper
    Ptr<WarperCreator> warperFactory = new cv::SphericalWarper();

    Ptr<RotationWarper> warper = warperFactory->create(static_cast<float>(warperScale)); 
	Ptr<RotationWarper> miniWarper = warperFactory->create(static_cast<float>(warperScale * workScale)); //Set scale to 1

	//Create data structures for warped images
	vector<Point> corners(n);
	vector<Mat> warpedImages(n);
	vector<Mat> warpedMasks(n);
	vector<Size> warpedSizes(n);

	vector<Point> miniCorners(n);
	vector<UMat> miniWarpedImages(n);
	vector<UMat> miniWarpedMasks(n);
	vector<Size> miniWarpedSizes(n);
	vector<UMat> miniWarpedImagesAsFloat(n);

	for (size_t i = 0; i < n; i++) {
       	Mat k = in[i]->intrinsics;

        //Big
        corners[i] = warper->warp(images[i], k, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
        warper->warp(masks[i], k, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
        warpedSizes[i] = warpedImages[i].size();
       
        //Mini      
        float miniKData[] = {
            k.at<float>(0, 0) * workScale, 0, k.at<float>(0, 2) * workScale,
            0, k.at<float>(1, 1) * workScale, k.at<float>(1, 2) * workScale,
            0, 0, 1
        };

        Mat miniK(3, 3, CV_32F, miniKData);

        miniCorners[i] = miniWarper->warp(miniImages[i], miniK, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, miniWarpedImages[i]);
        miniWarper->warp(miniMasks[i], miniK, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, miniWarpedMasks[i]);
        miniWarpedSizes[i] = miniWarpedImages[i].size(); 
        miniWarpedImages[i].convertTo(miniWarpedImagesAsFloat[i], CV_32F);
    }

	//Exposure compensation
    if(compensate) {
	 	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	    compensator->feed(miniCorners, miniWarpedImages, miniWarpedMasks);

	    for (size_t i = 0; i < n; i++)
	    {
	        compensator->apply(i, corners[i], warpedImages[i], warpedMasks[i]);
	    }
    }

    //Seam finding
    if(seam) {
		Ptr<SeamFinder> seamFinder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
		seamFinder->find(miniWarpedImagesAsFloat, miniCorners, miniWarpedMasks);
		
		//Merge masks from warper/seam finder
		Mat diliatedMask;
	    for (size_t i = 0; i < n; i++)
	    {
	        //Diliate mask (sharpen edges)
	        dilate(miniWarpedMasks[i], diliatedMask, Mat());
	        //Scale up mini mask to fit big mask. 
	        Mat seamedMask; 
	        resize(diliatedMask, seamedMask, warpedMasks[i].size());
	        warpedMasks[i] = warpedMasks[i] & seamedMask;
	    }
    }

    miniWarpedImagesAsFloat.clear();
    miniWarpedImages.clear();
    masks.clear();

    miniCorners.clear();
    miniImages.clear();
    miniMasks.clear();

    //Final blending
	Mat warpedImageAsShort;
	Ptr<Blender> blender;

	blender = Blender::createDefault(Blender::MULTI_BAND, true);
    Size destinationSize = resultRoi(corners, warpedSizes).size();
  
    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
    blender->prepare(corners, warpedSizes);

	for (size_t i = 0; i < n; i++)
	{
        warpedImages[i].convertTo(warpedImageAsShort, CV_16S);
		blender->feed(warpedImageAsShort, warpedMasks[i], corners[i]);
	}

	StitchingResult *res = new StitchingResult();
	blender->blend(res->image, res->mask);

	//Rotate by 180°
	//flip(res->image, res->image, -1);
	//flip(res->mask, res->mask, -1);

    res->corners = corners;
    res->sizes = warpedSizes;

    res->corner.x = corners[0].x;
    res->corner.y = corners[0].y;

    for(size_t i = 1; i < n; i++) {
        res->corner.x = min(res->corner.x, corners[i].x);
        res->corner.y = min(res->corner.y, corners[i].y);
    }

	return res;
}
}