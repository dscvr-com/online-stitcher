#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "support.hpp"
#include "simpleSeamer.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
    
void RStitcher::PrepareMatrices(const vector<InputImageP> &r) {

    //Orient around first image (Correct orientation from start.)
    vector<Mat> matrices(r.size());

    for(size_t i = 0; i <  r.size(); i++) {
        From4DoubleTo3Float(r[i]->adjustedExtrinsics, matrices[i]);
    }

    //Do wave correction
    waveCorrect(matrices, WAVE_CORRECT_HORIZ);

    for(size_t i = 0; i <  r.size(); i++) {
        From3FloatTo4Double(matrices[i], r[i]->adjustedExtrinsics);
    }
}

StitchingResultP RStitcher::Stitch(const std::vector<InputImageP> &in, const ExposureCompensator &exposure, ProgressCallback &progress, double ev, bool debug, const std::string&) {

    //This is needed because xcode does not like the CV stitching header.
    //So we can't initialize this constant in the header. 
    if(blendMode == -1) {
        blendMode = cv::detail::Blender::FEATHER;
    }
    
	size_t n = in.size();
    assert(n > 0);

	vector<Point> corners(n);
	vector<Size> warpedSizes(n);
    vector<Image> masks;
    vector<Image> images;
    vector<cv::detail::CameraParams> cameras;
    vector<Mat> intrinsicsList;
    
    masks.reserve(n);
    images.reserve(n);
    
    Ptr<WarperCreator> warperFactory = new cv::SphericalWarper();
    Ptr<RotationWarper> warper = warperFactory->create(static_cast<float>(warperScale));
    
    ProgressCallbackAccumulator stageProgress(progress, {0.1, 0.9});

    for(size_t i = 0; i < n; i++) {
        stageProgress.At(0)((float)i / (float)n);
        //Camera
        InputImageP img = in[i];
        
        cv::detail::CameraParams camera;
        From4DoubleTo3Float(img->adjustedExtrinsics, camera.R);
        Mat K; 
        Mat scaledIntrinsics;
        ScaleIntrinsicsToImage(img->intrinsics, img->image, scaledIntrinsics, debug ? 8 : 1);
        From3DoubleTo3Float(scaledIntrinsics, K);

        cameras.push_back(camera);
        intrinsicsList.push_back(K);

        //Warping
        Rect roi = warper->warpRoi(img->image.size(), K, camera.R);

        corners[i] = Point(roi.x, roi.y);
        warpedSizes[i] = Size(roi.width, roi.height);

    }
    stageProgress.At(0)(1);

    //Blending
    Ptr<Blender> blender;
	blender = Blender::createDefault(blendMode, false);
    blender->prepare(corners, warpedSizes);

    for(size_t i = 0; i < n; i++) {
        stageProgress.At(1)((float)i / (float)n);
        
        InputImageP img = in[i];
        if(!img->image.IsLoaded()) {
            img->image.Load();
        }
        
        const cv::detail::CameraParams &camera = cameras[i];
        const Mat &K = intrinsicsList[i];

        //Mask Warping
        Mat warpedMask;
        Mat mask(img->image.rows, img->image.cols, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, camera.R, INTER_NEAREST, BORDER_CONSTANT, warpedMask);
        mask.release();
           
        //Add 1px border to mask, to enable feather blending 
        warpedMask(Rect(0, 0, 1, warpedMask.rows)).setTo(Scalar::all(0));
        warpedMask(Rect(warpedMask.cols - 1, 0, 1, warpedMask.rows)).setTo(Scalar::all(0));
        //Image Warping
        Mat warpedImage;
        auto corner = warper->warp(img->image.data, K, camera.R, INTER_LINEAR, BORDER_CONSTANT, warpedImage);
        assert(corner == corners[i]);
        assert(warpedSizes[i] == warpedImage.size());
        
        img->image.Unload(); 
	    
        Mat warpedImageAsShort;
        warpedImage.convertTo(warpedImageAsShort, CV_16S);
        warpedImage.release();
        
        //Exposure compensate (Could move after warping?)
        exposure.Apply(warpedImageAsShort, in[i]->id, ev);

		blender->feed(warpedImageAsShort, warpedMask, corners[i]);

        warpedImageAsShort.release();
    }
    
    warper.release();
    warperFactory.release();


    cout << "Start blending" << endl;

	StitchingResultP res(new StitchingResult());
    {
        Mat resImage, resMask;
        blender->blend(resImage, resMask);

        resImage.convertTo(resImage, CV_8U);

        res->image = Image(resImage);
        res->mask = Image(resMask);
    }
    blender.release();

    res->corner.x = corners[0].x;
    res->corner.y = corners[0].y;

    for(size_t i = 1; i < n; i++) {
        res->corner.x = min(res->corner.x, corners[i].x);
        res->corner.y = min(res->corner.y, corners[i].y);
    }
    stageProgress.At(1)(1);

	return res;
}
}
