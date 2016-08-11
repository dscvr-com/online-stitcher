#include <vector>
#include <deque>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching.hpp>

#include "../math/support.hpp"
#include "../common/image.hpp"
#include "../common/support.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/static_timer.hpp"
#include "ringStitcher.hpp"
#include "dynamicSeamer.hpp"
#include "flowBlender.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

static const bool debug = false;

namespace optonaut {

struct FlowImage {
    Point corner;
    Mat image;
    Mat flow;
};

typedef shared_ptr<FlowImage> FlowImageP;
    
void RingStitcher::PrepareMatrices(const vector<InputImageP> &r) {

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

class AsyncRingStitcher::Impl {
private:
    size_t n;
    
    //cv::Ptr<cv::detail::Blender> blender;
    cv::Ptr<cv::detail::RotationWarper> warper;
    cv::Ptr<cv::WarperCreator> warperFactory;
    std::vector<cv::Point> corners;
    std::vector<cv::Size> warpedSizes;
    cv::UMat uxmap, uymap;
    cv::Rect resultRoi;
    cv::Rect dstRoi;
    cv::Rect coreRoi;
    RingProcessor<FlowImageP> queue;
    cv::Mat warpedMask;
    cv::Mat K;

    FlowBlender blender;

    cv::Size initialSize;

    //Stitcher feed function.
    void Feed(const FlowImageP &in) {
        STimer feedTimer;

        if(debug) {
            imwrite("dbg/feed_" + ToString(in->id) + ".jpg", in->image.data);
            imwrite("dbg/feed_mask_" + ToString(in->id) + ".jpg", in->mask.data);
        }

        //Mat warpedImageAsShort;
        //in->image.data.convertTo(warpedImageAsShort, CV_16S);

        Rect imageRoi(in->corner, in->image.size());
        Rect overlap = imageRoi & resultRoi;

        Rect overlapI(0, 0, overlap.width, overlap.height);
        if(overlap.width == imageRoi.width) {
            //Image fits.
            blender.Feed(in->image(overlapI), in->flow(overlapI), in->corner);
        } else {
            //Image overlaps on X-Axis and we have to blend two parts.
            blender.Feed(in->image(overlapI), 
                    in->flow(overlapI), 
                    overlap.tl());

            Rect other(resultRoi.x, overlap.y, 
                    imageRoi.width - overlap.width, overlap.height);
            Rect otherI(overlap.width, 0, other.width, other.height);
            blender.Feed(in->image(otherI), 
                    in->flow(otherI), 
                    other.tl());
        }

        //warpedImageAsShort.release();

        feedTimer.Tick("Image Fed");
    }

    //Seam finder function. 
    void FindSeams(const FlowImageP &a,
            const FlowImageP &b) {

        Rect aRoi(a->corner, a->image.size());
        Rect bRoi(b->corner, b->image.size());

        Rect overlap = aRoi & bRoi;

        Rect aOverlap(overlap.tl() - a.corner, overlap.size());
        Rect bOverlap(overlap.tl() - b.corner, overlap.size());



        // Nope, no seams, we use flow blending. 
        return;
    };
    public:
    Impl(
            const InputImageP img, vector<Mat> rotations,
            float warperScale, bool, int) :
        queue(1, 
            std::bind(&Impl::FindSeams, this,
                std::placeholders::_1, std::placeholders::_2), 
            std::bind(&Impl::Feed, this, std::placeholders::_1)) {
        STimer timer; 
        timer.Tick("Async Preperation");
        
        n = rotations.size();
        
        AssertGT(n, (size_t)0);

        corners.resize(n);
        warpedSizes.resize(n);

        warperFactory = new cv::SphericalWarper();
        warper = warperFactory->create(static_cast<float>(warperScale));

        Mat scaledK;
        ScaleIntrinsicsToImage(img->intrinsics, img->image, scaledK);
        From3DoubleTo3Float(scaledK, K);

        initialSize = img->image.size();

        // Calulate result ROI
        for(size_t i = 0; i < n; i++) {
            Mat R;
            From3DoubleTo3Float(rotations[i], R); 
            //Warping
            Rect roi = warper->warpRoi(img->image.size(), K, R);
            corners[i] = Point(roi.x, roi.y);
            warpedSizes[i] = Size(roi.width, roi.height);

            rotations[i] = R;
        }

        resultRoi = cv::detail::resultRoi(corners, warpedSizes);
        resultRoi = Rect(resultRoi.x, resultRoi.y,
                         resultRoi.width, 
                         resultRoi.height);
        blender.Prepare(resultRoi);

        //Prepare global masks and distortions. 
        const Mat &R = rotations[0];

        dstRoi = GetOuterRectangle(*warper, K, R, img->image.size());
        coreRoi = GetInnerRectangle(*warper, K, R, img->image.size());
        coreRoi = Rect(coreRoi.x + 1, coreRoi.y + 1, coreRoi.width - 1, coreRoi.height - 1);
        
        cout << dstRoi << endl;
        cout << coreRoi << endl;

        warper->buildMaps(img->image.size(), K, R, uxmap, uymap);
        coreRoi = Rect(coreRoi.tl() - dstRoi.tl(), coreRoi.size() - Size(1, 1));

        warpedMask = Mat(dstRoi.size(), CV_8U, Scalar::all(0));
        {
            Mat mask = Mat(img->image.rows, img->image.cols, CV_8U, Scalar::all(255));
            remap(mask, warpedMask, uxmap, uymap, INTER_NEAREST, BORDER_CONSTANT); 
            warpedMask = warpedMask(coreRoi);
            mask.release();
        }
    }

    void Push(const InputImageP img) {
        
        Assert(img != nullptr);

        // We assume equal dimensions and intrinsics for a performance gain. 
        //AssertEQM(img->image.size(), initialSize, "Image has same dimensions as initial image");
        
        STimer detailTimer;

        bool autoUnload = false;
        if(!img->image.IsLoaded()) {
            img->image.Load();
            autoUnload = true;
        }
        detailTimer.Tick("Image Loaded");
        
        FlowImageP res(new FlowImage()); 

        Mat R;
        From3DoubleTo3Float(img->adjustedExtrinsics, R);
        
        //Image Warping
        Mat warpedImage(dstRoi.size(), CV_8UC3);
        remap(img->image.data, warpedImage, uxmap, uymap, 
                INTER_LINEAR, BORDER_CONSTANT); 
        res->image = Image(warpedImage(coreRoi));
        res->id = img->id;
      
        //Calculate Image Position (without wrapping around)
        Rect roi = GetInnerRectangle(*warper, K, R, img->image.size());
        Point bl = roi.tl() - Point(0, roi.height);
        Point tl = roi.tl();

        Point cornerLeft;

        if(abs(bl.x - tl.x) > roi.width / 2) {
            //Corner case. Difference between left corners
            //is more than half the image. 
            if(bl.x < tl.x) {
                cornerLeft = tl;
            } else {
                cornerLeft = bl;
            }
        } else {
            //Standard case. 
            if(bl.x > tl.x) {
                cornerLeft = tl;
            } else {
                cornerLeft = bl;
            }
        }

        res->corner = Point(cornerLeft.x, roi.y);
       
        if(autoUnload) { 
            img->image.Unload(); 
        }

        detailTimer.Tick("Image Warped");
        queue.Push(res);
        detailTimer.Tick("Image Seamed And and Fed");
    }

    StitchingResultP Finalize() {
        queue.Flush();

        STimer timer;

        StitchingResultP res(new StitchingResult());
        {
            Mat resImage, resMask;
            //blender.Blend(resImage, resMask);

            //if(resImage.type() != CV_8UC3) {
            //    resImage.convertTo(resImage, CV_8UC3);
            //}

            res->image = Image(blender.GetResult());
            res->mask = Image(blender.GetResultMask());
        }
        //blender.release();
        warper.release();
        warperFactory.release();

        res->corner.x = corners[0].x;
        res->corner.y = corners[0].y;
        res->seamed = false;

        for(size_t i = 1; i < n; i++) {
            res->corner.x = min(res->corner.x, corners[i].x);
            res->corner.y = min(res->corner.y, corners[i].y);
        }
        timer.Tick("Ring Stitching Blended");

        return res;
    }
};

AsyncRingStitcher::AsyncRingStitcher(const InputImageP firstImage,
        std::vector<cv::Mat> rotations, float warperScale,
        bool fast, int roiBuffer) :
    pimpl_(new Impl(firstImage, rotations, warperScale, fast, roiBuffer)),
    warperScale(warperScale) {
        AssertFalseInProduction(debug);        
}

void AsyncRingStitcher::Push(const InputImageP image) { pimpl_->Push(image); }

StitchingResultP AsyncRingStitcher::Finalize() { return pimpl_->Finalize(); }

AsyncRingStitcher::~AsyncRingStitcher() {
    delete pimpl_;
}
}
