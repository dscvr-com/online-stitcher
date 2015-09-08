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
    
void RStitcher::PrepareMatrices(const vector<ImageP> &r) {


    //Orient around first image (Correct orientation from start.)
    Mat center = r[0]->adjustedExtrinsics.inv();
    vector<Mat> matrices(r.size());

    for(size_t i = 0; i <  r.size(); i++) {
        From4DoubleTo3Float(center * r[i]->adjustedExtrinsics, matrices[i]);
    }

    //Do wave correction
    waveCorrect(matrices, WAVE_CORRECT_HORIZ);

    for(size_t i = 0; i <  r.size(); i++) {
        From3FloatTo4Double(matrices[i], r[i]->adjustedExtrinsics);
    }
}

StitchingResultP RStitcher::Stitch(const std::vector<ImageP> &in, bool debug) {

    const int maskOffset = 20000;
    const int imageOffset = 30000;

    //This is needed because xcode does not like the CV stitching header.
    //So we can't initialize this constant in the header. 
    if(blendMode == -1) {
        blendMode = cv::detail::Blender::FEATHER;
    }
    
	size_t n = in.size();
    assert(n > 0);

	vector<Point> corners(n);
	vector<Size> warpedSizes(n);
    
    Ptr<WarperCreator> warperFactory = new cv::SphericalWarper();
    Ptr<RotationWarper> warper = warperFactory->create(static_cast<float>(warperScale)); 

    for(size_t i = 0; i < n; i++) {
        
        //Camera
        ImageP img = in[i];
        if(!img->IsLoaded())
            img->LoadFromDisk();
        cv::detail::CameraParams camera;
        From4DoubleTo3Float(img->adjustedExtrinsics, camera.R);
        Mat K; 
        Mat scaledIntrinsics;

        ScaleIntrinsicsToImage(img->intrinsics, img->img, scaledIntrinsics, debug ? 10 : 1);
        From3DoubleTo3Float(scaledIntrinsics, K);

        //Mask
        Mat mask(img->img.rows, img->img.cols, CV_8U);
        mask.setTo(Scalar::all(255));

        //Warping
        Mat warpedMask;
        Mat warpedImage;

        corners[i] = warper->warp(img->img, K, camera.R, INTER_LINEAR, BORDER_CONSTANT, warpedImage);
        img->Unload();
        warper->warp(mask, K, camera.R, INTER_NEAREST, BORDER_CONSTANT, warpedMask);
        mask.release();
        warpedSizes[i] = warpedImage.size();

        Image::SaveToDisk(i + maskOffset, warpedMask);
        warpedMask.release();
        Image::SaveToDisk(i + imageOffset, warpedImage);
        warpedImage.release();

    }

    warper.release();
    warperFactory.release();

    //Blending
    Ptr<Blender> blender;
	blender = Blender::createDefault(blendMode, true);
    blender->prepare(corners, warpedSizes);

    for(size_t i = 0; i < n; i++) {
        Mat warpedMask;
        Mat warpedImage;

        Image::LoadFromDisk(i + maskOffset, warpedMask, CV_LOAD_IMAGE_GRAYSCALE);
        Image::LoadFromDisk(i + imageOffset, warpedImage);

	    Mat warpedImageAsShort;
        warpedImage.convertTo(warpedImageAsShort, CV_16S);
		blender->feed(warpedImageAsShort, warpedMask, corners[i]);

        warpedMask.release();
        warpedImage.release();
        warpedImageAsShort.release();

    }

    cout << "Start blending" << endl;

	StitchingResultP res(new StitchingResult());
	blender->blend(res->image, res->mask);

    blender.release();

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
