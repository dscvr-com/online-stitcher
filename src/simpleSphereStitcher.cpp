#include <vector>
#include <deque>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "support.hpp"
#include "simpleSeamer.hpp"
#include "verticalDynamicSeamer.hpp"
#include "ringProcessor.hpp"

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
    Rect resultRoi = cv::detail::resultRoi(corners, warpedSizes);
    blender->prepare(resultRoi);

    auto findSeams = [] (StitchingResultP &a, StitchingResultP &b) {
            VerticalDynamicSeamer::Find(a->image.data, b->image.data, 
                    a->mask.data, b->mask.data, 
                    a->corner, b->corner, 5, a->id);
    };

    auto feed = [&] (StitchingResultP &in) {
            Mat warpedImageAsShort;
            in->image.data.convertTo(warpedImageAsShort, CV_16S);
            exposure.Apply(warpedImageAsShort, in->id, ev);

            Rect imageRoi(in->corner, in->image.size());
            Rect overlap = imageRoi & resultRoi;

            if(overlap == imageRoi) {
                blender->feed(warpedImageAsShort, in->mask.data, in->corner);
            } else {
                Rect overlapI(0, 0, overlap.width, overlap.height);
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
    };

    UMat uxmap, uymap;
    const Mat &K = intrinsicsList[0];
    const Mat &R = cameras[0].R;
    Rect dstRoi = warper->buildMaps(in[0]->image.size(), K, R, uxmap, uymap);
    dstRoi = Rect(dstRoi.x, dstRoi.y, dstRoi.width + 1, dstRoi.height + 1);

    RingProcessor<StitchingResultP> queue(1, findSeams, feed);

    for(size_t i = 0; i < n; i++) {
        stageProgress.At(1)((float)i / (float)n);
        
        InputImageP img = in[i];
        if(!img->image.IsLoaded()) {
            img->image.Load();
        }
        
        StitchingResultP res(new StitchingResult()); 

        //Mask Warping
        Mat mask(img->image.rows, img->image.cols, CV_8U);
        mask.setTo(Scalar::all(255));
        Mat warpedMask(dstRoi.size(), CV_8U);
        remap(mask, warpedMask, uxmap, uymap, INTER_NEAREST, BORDER_CONSTANT); 
        mask.release();
        res->mask = Image(warpedMask);
        
           
        //Image Warping
        Mat warpedImage(dstRoi.size(), CV_8UC3);
        remap(img->image.data, warpedImage, uxmap, uymap, INTER_LINEAR, BORDER_CONSTANT); 
        res->image = Image(warpedImage);

        res->id = img->id;
        Point cornerTop = warper->warpPoint(Point(img->image.cols / -2, img->image.rows / -2), intrinsicsList[i], cameras[i].R);
        //Point cornerBot = warper->warpPoint(Point(img->image.cols / -2, img->image.rows / -2), intrinsicsList[i], cameras[i].R);
        res->corner = Point(cornerTop.x, corners[i].y);
        
        img->image.Unload(); 

        queue.Push(res);
    }

    queue.Flush();
    
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
