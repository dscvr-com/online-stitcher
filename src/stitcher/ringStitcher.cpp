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

using namespace std;
using namespace cv;
using namespace cv::detail;

static const bool debug = true;

namespace optonaut {
    
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
    
    cv::Ptr<cv::detail::Blender> blender;
    cv::Ptr<cv::detail::RotationWarper> warper;
    cv::Ptr<cv::WarperCreator> warperFactory;
    std::vector<cv::Point> corners;
    std::vector<cv::Size> warpedSizes;
    cv::UMat uxmap, uymap;
    cv::Rect resultRoi;
    cv::Rect dstRoi;
    cv::Rect dstCoreMaskRoi;
    RingProcessor<StitchingResultP> queue;
    cv::Mat warpedMask;
    cv::Mat K;
    bool fast;

    cv::Size initialSize;

    //Stitcher feed function.
    void Feed(const StitchingResultP &in) {
        STimer feedTimer;

        if(debug) {
            imwrite("dbg/feed_" + ToString(in->id) + ".jpg", in->image.data);
            imwrite("dbg/feed_mask_" + ToString(in->id) + ".jpg", in->mask.data);
        }

        Mat warpedImageAsShort;
        in->image.data.convertTo(warpedImageAsShort, CV_16S);

        Rect imageRoi(in->corner, in->image.size());
        Rect overlap = imageRoi & resultRoi;

        Rect overlapI(0, 0, overlap.width, overlap.height);
        cout << "warpedImageAsShort size " << warpedImageAsShort.size() << endl;
        cout << "in->corner " << in->corner << endl;
        cout << "imageRoi " << imageRoi << endl;
        cout << "resultRoi " << resultRoi << endl;
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
    }

    //Seam finder function. 
    void FindSeams(const StitchingResultP &a,
            const StitchingResultP &b) {

        // Nope, no seams.
        if(fast) {
            return;
        }

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
                b->corner + dstCoreMaskRoi.tl(), 0, 1, a->id);
    };
    public:
    Impl(
            const InputImageP img, vector<Mat> rotations,
            float warperScale, bool fast, int roiBuffer) :
        queue(1, 
            std::bind(&Impl::FindSeams, this,
                std::placeholders::_1, std::placeholders::_2), 
            std::bind(&Impl::Feed, this, std::placeholders::_1)),
        fast(fast) {
        
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
            cout << "Rect Roi " << roi << endl; 
          //  cout << "[init] ImageSource" << rotations[i]->image.source << endl;
           // cout << "[init] ImageSize" << rotations[i]->image.size() << endl;
            corners[i] = Point(roi.x, roi.y);
            warpedSizes[i] = Size(roi.width, roi.height);

            rotations[i] = R;
        }

        //Blending
        if(fast) {
            blender = Blender::createDefault(cv::detail::Blender::FEATHER, false);
        } else {
            MultiBandBlender* mb;
            blender = Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
            mb = dynamic_cast<MultiBandBlender*>(blender.get());
            mb->setNumBands(5);
        }

        resultRoi = cv::detail::resultRoi(corners, warpedSizes);
        resultRoi = Rect(resultRoi.x - roiBuffer, resultRoi.y - roiBuffer,
                         resultRoi.width + roiBuffer * 2, 
                         resultRoi.height + roiBuffer * 2);
        blender->prepare(resultRoi);

        //Prepare global masks and distortions. 
        const Mat &R = rotations[0];
        Rect coreMaskRoi = Rect(img->image.cols / 4, 0, 
                img->image.cols / 2, img->image.rows);
        dstRoi = warper->buildMaps(img->image.size(), K, R, uxmap, uymap);
        dstCoreMaskRoi = warper->warpRoi(coreMaskRoi.size(), K, R);
        dstCoreMaskRoi = Rect((dstRoi.width - dstCoreMaskRoi.width) / 2, 
                              0, dstCoreMaskRoi.width, dstCoreMaskRoi.height); 
        dstRoi = Rect(dstRoi.x, dstRoi.y, dstRoi.width + 1, dstRoi.height + 1);

        warpedMask = Mat(dstRoi.size(), CV_8U);
        {
            //Mask Warping - we force a tiny mask for each image.
            Mat mask = Mat::zeros(img->image.rows, img->image.cols, CV_8U);
            if(fast) {
                mask.setTo(Scalar::all(255));
            } else {
                mask(coreMaskRoi).setTo(Scalar::all(255));
            }
            remap(mask, warpedMask, uxmap, uymap, INTER_NEAREST, BORDER_CONSTANT); 
            mask.release();
        }
    }

    void Push(const InputImageP img) {
        
        Assert(img != nullptr);

        // We assume equal dimensions and intrinsics for a performance gain. 
        AssertEQM(img->image.size(), initialSize, "Image has same dimensions as initial image");
        
        STimer detailTimer;

        bool autoUnload = false;
        cout << "[push] ImageSource" << img->image.source << endl; 
        cout << "[push] ImageSize" << img->image.size() << endl; 
        if(!img->image.IsLoaded()) {
            img->image.Load();
            autoUnload = true;
        }
        detailTimer.Tick("Image Loaded");
        
        StitchingResultP res(new StitchingResult()); 

        res->mask = Image(warpedMask.clone());

        Mat R;
        From3DoubleTo3Float(img->adjustedExtrinsics, R);
        
        //Image Warping
        Mat warpedImage(dstRoi.size(), CV_8UC3);
        remap(img->image.data, warpedImage, uxmap, uymap, 
                INTER_LINEAR, BORDER_CONSTANT); 
        res->image = Image(warpedImage);
        res->id = img->id;
      
        /*
            Mat rvec;
            ExtractRotationVector( img->adjustedExtrinsics , rvec);
            cout << "## Rotation X: " << rvec.at<double>(0) << endl;
            cout << "## Rotation Y: " << rvec.at<double>(1) << endl;
            cout << "## Rotation Z: " << rvec.at<double>(2) << endl;
            cout << "## rvec : " << rvec << endl;
        */


        //Calculate Image Position (without wrapping around)
        Point tl = warper->warpPoint(Point(0, img->image.rows), K, R);
        
        // cout << "Point tl " << tl << endl;
        // cout << "K matrix " << K << endl;
        // cout << "img->adjustedExtrinsics " << img->adjustedExtrinsics << endl;


        Point bl = warper->warpPoint(Point(0, 0), K, R);
        Rect roi = warper->warpRoi(img->image.size(), K, R);
        Size fullRoi = roi.size();

        // cout << "bl" << bl << endl;
        // cout << "roi" << roi << endl;
        // cout << "fullRoi" << fullRoi << endl;


        Point cornerLeft;

        if(abs(bl.x - tl.x) > fullRoi.width / 2) {
            //Corner case. Difference between left corners
            //is more than half the image. 
            // cout << "corner case" << endl;
            if(bl.x < tl.x) {
                cornerLeft = tl;
            } else {
                cornerLeft = bl;
            }
        } else {
            //Standard case. 
            // cout << "standard case" << endl;
            if(bl.x > tl.x) {
                cornerLeft = tl;
            } else {
                cornerLeft = bl;
            }
        }

        res->corner = Point(cornerLeft.x, roi.y);
        cout << " res->corner " << res->corner << endl;
       
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
            blender->blend(resImage, resMask);

            if(resImage.type() != CV_8UC3) {
                resImage.convertTo(resImage, CV_8UC3);
            }

            res->image = Image(resImage);
            res->mask = Image(resMask);
        }
        blender.release();
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
    pimpl_(new Impl(firstImage, rotations, warperScale, fast, roiBuffer)) {
        
}

void AsyncRingStitcher::Push(const InputImageP image) { pimpl_->Push(image); }

StitchingResultP AsyncRingStitcher::Finalize() { return pimpl_->Finalize(); }

AsyncRingStitcher::~AsyncRingStitcher() {
    delete pimpl_;
}
}
