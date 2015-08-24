#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "support.hpp"
#include "simpleSeamer.hpp"
#include "thresholdSeamer.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
vector<ImageP> RStitcher::PrepareMatrices(vector<ImageP> r) {


    //Orient around first image (Correct orientation from start.)
    Mat center = r[0]->extrinsics.inv();
    vector<Mat> matrices(r.size());

    for(size_t i = 0; i <  r.size(); i++) {
        From4DoubleTo3Float(center * r[i]->extrinsics, matrices[i]);
    }

    //Do wave correction
    waveCorrect(matrices, WAVE_CORRECT_HORIZ);

    for(size_t i = 0; i <  r.size(); i++) {
        From3FloatTo4Double(matrices[i], r[i]->extrinsics);
    }

    return r;
}

StitchingResultP RStitcher::Stitch(std::vector<ImageP> in, bool debug) {
	size_t n = in.size();
    assert(n > 0);

    vector<Mat> images(n);
    vector<cv::detail::CameraParams> cameras(n);

    for(size_t i = 0; i < n; i++) {
        images[i] = in[i]->img;
        From4DoubleTo3Float(in[i]->extrinsics, cameras[i].R);
    }

	//Create masks
    vector<Mat> masks(n);

    for(size_t i = 0; i < n; i++) {
        masks[i].create(images[i].size(), CV_8U);
        ThresholdSeamer::createMask(masks[i]);
        masks[i].setTo(Scalar::all(255));
        //imwrite("dbg/premask" + ToString(i) + ".jpg", masks[i]);
    }

    //Create warper
    Ptr<WarperCreator> warperFactory = new cv::SphericalWarper();
    Ptr<RotationWarper> warper = warperFactory->create(static_cast<float>(warperScale)); 

	//Create data structures for warped images
	vector<Point> corners(n);
	vector<UMat> warpedImages(n);
    vector<UMat> warpedImagesAsFloat(n);
	vector<UMat> warpedMasks(n);
	vector<Size> warpedSizes(n);

	for (size_t i = 0; i < n; i++) {
       	Mat k;
        Mat scaledIntrinsics;
        ScaleIntrinsicsToImage(in[i]->intrinsics, images[i], scaledIntrinsics, debug ? 10 : 1);
        From3DoubleTo3Float(scaledIntrinsics, k);

        //Big
        corners[i] = warper->warp(images[i], k, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, warpedImages[i]);
        warper->warp(masks[i], k, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
        warpedSizes[i] = warpedImages[i].size();
        warpedImages[i].convertTo(warpedImagesAsFloat[i], CV_32F);
        //imwrite("dbg/warpmask" + ToString(i) + ".jpg", warpedMasks[i]);

    }
    masks.clear();
    images.clear();

    //Seam finding
    if(seam) {
        //Ptr<SeamFinder> seamFinder = new GraphCutSeamFinder();
		Ptr<SeamFinder> seamFinder = new ThresholdSeamer();
		seamFinder->find(warpedImagesAsFloat, corners, warpedMasks);
    }
    warpedImagesAsFloat.clear();

    for (size_t i = 0; i < n; i++) {
        //imwrite("dbg/submask" + ToString(i) + ".jpg", warpedMasks[i]);
        ThresholdSeamer::brightenMask(warpedMasks[i]);
        //imwrite("dbg/mask" + ToString(i) + ".jpg", warpedMasks[i]);
    }

    //Final blending
	Mat warpedImageAsShort;
	Ptr<Blender> blender;

	blender = Blender::createDefault(blendMode, true);
    //Size destinationSize = resultRoi(corners, warpedSizes).size();
  
    blender->prepare(corners, warpedSizes);

	for (size_t i = 0; i < n; i++)
	{
        warpedImages[i].convertTo(warpedImageAsShort, CV_16S);
		blender->feed(warpedImageAsShort, warpedMasks[i], corners[i]);
	}

	StitchingResultP res(new StitchingResult());
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
    
    res->image.convertTo(res->image, CV_8U);

	return res;
}
}
