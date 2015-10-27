#include <vector>
#include <deque>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "support.hpp"
#include "simpleSeamer.hpp"
#include "dynamicSeamer.hpp"
#include "ringProcessor.hpp"
#include "static_timer.hpp"

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

    STimer timer; 
    timer.Tick("Ring Stitching Start");

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
    MultiBandBlender* mb;
	blender = Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
    mb = dynamic_cast<MultiBandBlender*>(blender.get());
    mb->setNumBands(6);
    Rect resultRoi = cv::detail::resultRoi(corners, warpedSizes);
    blender->prepare(resultRoi);
    
    //Prepare global masks and distortions. 
    UMat uxmap, uymap;
    const Mat &K = intrinsicsList[0];
    const Mat &R = cameras[0].R;
    Rect coreMaskRoi = Rect(in[0]->image.cols / 4, 0, in[0]->image.cols / 2, in[0]->image.rows);
    Rect dstRoi = warper->buildMaps(in[0]->image.size(), K, R, uxmap, uymap);
    Rect dstCoreMaskRoi = warper->warpRoi(coreMaskRoi.size(), K, R);
    dstCoreMaskRoi = Rect((dstRoi.width - dstCoreMaskRoi.width) / 2, 
                          0, dstCoreMaskRoi.width, dstCoreMaskRoi.height); 
    dstRoi = Rect(dstRoi.x, dstRoi.y, dstRoi.width + 1, dstRoi.height + 1);


    Mat warpedMask(dstRoi.size(), CV_8U);
    {
        //Mask Warping - we force a tiny mask for each image.
        InputImageP img = in[0];
        Mat mask = Mat::zeros(img->image.rows, img->image.cols, CV_8U);
        mask(coreMaskRoi).setTo(Scalar::all(255));
        remap(mask, warpedMask, uxmap, uymap, INTER_NEAREST, BORDER_CONSTANT); 
        mask.release();
        
    }


    //Seam finder function. 
    auto findSeams = [&] (StitchingResultP &a, StitchingResultP &b) {

            Point aCorner = a->corner;

            if(aCorner.x > b->corner.x) {
                //Warp a around if b is already warped around.  
                aCorner.x -= resultRoi.width;
            }

            //We can boost performance here if we omit the
            //black regions of our masks. 
            
            Mat aImg = a->image.data(dstCoreMaskRoi);
            Mat bImg = b->image.data(dstCoreMaskRoi);
            Mat aMask = a->mask.data(dstCoreMaskRoi);
            Mat bMask = b->mask.data(dstCoreMaskRoi);

            DynamicSeamer::Find<true>(aImg, bImg, 
                    aMask, bMask, 
                    aCorner + dstCoreMaskRoi.tl(),
                    b->corner + dstCoreMaskRoi.tl(), 1, a->id);
    };

    //Stitcher feed function. 
    auto feed = [&] (StitchingResultP &in) {

            STimer feedTimer;

            Mat warpedImageAsShort;
            in->image.data.convertTo(warpedImageAsShort, CV_16S);
            exposure.Apply(warpedImageAsShort, in->id, ev);

            Rect imageRoi(in->corner, in->image.size());
            Rect overlap = imageRoi & resultRoi;
        
            Rect overlapI(0, 0, overlap.width, overlap.height);
        
            if(overlap.width == imageRoi.width) {
                //Image fits.
                blender->feed(warpedImageAsShort(overlapI), in->mask.data(overlapI), in->corner);
            } else {
                //Image overlaps on X-Axis and we have to blend two parts.
                blender->feed(warpedImageAsShort(overlapI), 
                        in->mask.data(overlapI), 
                        overlap.tl());

                Rect other(resultRoi.x, overlap.y, 
                        imageRoi.width - overlap.width, overlap.height);
                Rect otherI(overlap.width, 0, other.width, other.height);
                blender->feed(warpedImageAsShort(otherI), 
                        in->mask.data(otherI), 
                        other.tl());
            }
            warpedImageAsShort.release();

            feedTimer.Tick("Image Fed");
    };

    RingProcessor<StitchingResultP> queue(1, findSeams, feed);

    STimer detailTimer;

    for(size_t i = 0; i < n; i++) {
        stageProgress.At(1)((float)i / (float)n);
        
        InputImageP img = in[i];
        if(!img->image.IsLoaded()) {
            img->image.Load();
        }
        detailTimer.Tick("Image Loaded");
        
        StitchingResultP res(new StitchingResult()); 

        res->mask = Image(warpedMask.clone());
        
        //Image Warping
        Mat warpedImage(dstRoi.size(), CV_8UC3);
        remap(img->image.data, warpedImage, uxmap, uymap, 
                INTER_LINEAR, BORDER_CONSTANT); 
        res->image = Image(warpedImage);
        res->id = img->id;

        //Calculate Image Position (without wrapping aroing)
        Point cornerTop = warper->warpRoi(Size(1, 1), 
                intrinsicsList[i], cameras[i].R).tl();
        res->corner = Point(cornerTop.x, corners[i].y);
        
        img->image.Unload(); 
        detailTimer.Tick("Image Warped");
        queue.Push(res);
        detailTimer.Tick("Image Seamed And and Fed");
    }

    queue.Flush();
    timer.Tick("Ring Stitching Warped");
    
    warper.release();
    warperFactory.release();

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
    res->seamed = false;

    for(size_t i = 1; i < n; i++) {
        res->corner.x = min(res->corner.x, corners[i].x);
        res->corner.y = min(res->corner.y, corners[i].y);
    }
    stageProgress.At(1)(1);
    timer.Tick("Ring Stitching Blended");

	return res;
}
}
