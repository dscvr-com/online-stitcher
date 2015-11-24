#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
//define private public, so we can access
//opencv internal information and print it (to debug the multi-band blender ONLY)
#define private public
#include <opencv2/stitching.hpp>
#undef private

#include "../math/support.hpp"
#include "../common/image.hpp"
#include "../common/support.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/static_timer.hpp"
#include "../common/functional.hpp"
#include "../imgproc/correlation.hpp"
#include "ringStitcher.hpp"
#include "multiringStitcher.hpp"
#include "stitchingResult.hpp"
#include "dynamicSeamer.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
    
    STimer stitcherTimer;

    static void LoadSRP(const StitchingResultP &a) {
        a->image.Load();
        a->mask.Load(CV_LOAD_IMAGE_GRAYSCALE);
    }
    
    static void LoadSRPGrayscaleDropMask(const StitchingResultP &a) {
        a->image.Load(CV_LOAD_IMAGE_GRAYSCALE);
    }

    static void UnLoadSRP(const StitchingResultP &a) {
        a->image.Unload();
        a->mask.Unload();
    } 
    
    void UnLoadSRPStoreMask(CheckpointStore &store, const StitchingResultP &a) {
        a->image.Unload();
        a->seamed = true;
        cout << "Saving mask" << a->id << endl;
        store.SaveRingMask(a->id, a);
        a->mask.Unload();
    }
         
    void MultiRingStitcher::AdjustCorners(std::vector<StitchingResultP> &rings, ProgressCallback &progress) {

        auto correlate = [this] (const StitchingResultP &imgA, const StitchingResultP &imgB) {
            const int warp = MOTION_TRANSLATION;
            Mat affine = Mat::eye(2, 3, CV_32F);

            const int iterations = 100;
            const double eps = 1e-3;

            int dy = imgA->corner.y - imgB->corner.y;
            affine.at<float>(1, 2) = dy;

            TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS,
                    iterations, eps);

            try {
                findTransformECC(imgA->image.data, imgB->image.data, affine, warp, termination);
                dy = affine.at<float>(1, 2);
            } catch (Exception ex) {
                // :(
            }

            dyCache.push_back(dy);
        };

        store.LoadRingAdjustment(dyCache);

        if(dyCache.size() == 0) {
            RingProcessor<StitchingResultP> queue(1, 0, LoadSRPGrayscaleDropMask, correlate, UnLoadSRP);
            queue.Process(rings, progress);
            store.SaveRingAdjustment(dyCache);
        }

        for(size_t i = 1; i < rings.size(); i++) {
            rings[i]->corner.y = rings[i - 1]->corner.y - dyCache[i - 1];
        }
    }

    void FindSeams(std::vector<StitchingResultP> &rings, CheckpointStore &store) {
        RingProcessor<StitchingResultP> queue(1, 0, 
                LoadSRP, 
                DynamicSeamer::FindHorizontalFromStitchingResult, 
                std::bind(&UnLoadSRPStoreMask, store, std::placeholders::_1));

        auto ordered = fun::orderby<StitchingResultP, int>(rings, [](const StitchingResultP &x) {
            return (int)x->corner.x; 
        });
        queue.Process(rings);
    }
    
    void MultiRingStitcher::InitializeForStitching(std::vector<std::vector<InputImageP>> &rings, ExposureCompensator &exposure, double ev) {
        
        this->rings = rings;
        this->exposure.SetGains(exposure.GetGains());
        this->ev = ev;
        this->dyCache = vector<int>();
    }
    
    StitchingResultP MultiRingStitcher::StitchRing(const vector<InputImageP> &ring, ProgressCallback &progress, int ringId, bool debug, const string &debugName) const {
        
        cout << "Attempting to stitch ring " << ringId << endl;

        StitchingResultP res = store.LoadRing(ringId);

        if(res != NULL) {
            progress(1);
            return res;
        }
        
        RingStitcher stitcher(store);
        
        res = stitcher.Stitch(ring, exposure, progress, ev, debug, debugName);
        res->id = ringId;

        store.SaveRing(ringId, res);

        res->image.Unload();
        res->mask.Unload();

        return res;
    }

    StitchingResultP MultiRingStitcher::Stitch(ProgressCallback &progress, bool debug, const string &debugName) {

        stitcherTimer.Tick("StitchStart");
        
        StitchingResultP res = store.LoadOptograph();
        if(res != NULL) {
            //Caller expects a loaded image. 
            res->image.Load();
            progress(1);
            return res;
        }
        
        res = StitchingResultP(new StitchingResult());
    
        vector<float> weights(rings.size() + 2);
        fill(weights.begin(), weights.end(), 1.0f / weights.size());
        ProgressCallbackAccumulator progressCallbacks(progress, weights);
        ProgressCallback &ringAdjustmentProgress = progressCallbacks.At(weights.size() - 2);
        ProgressCallback &finalBlendingProgress = progressCallbacks.At(weights.size() - 1);

        vector<StitchingResultP> stitchedRings;
       
        cout << "Attempting to stitch rings." << endl;
        int margin = -1; 
        
        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) {
                progressCallbacks.At(i)(1);
                continue;
            }
            
            auto res = StitchRing(rings[i], progressCallbacks.At(i), (int)i, debug, debugName);
            
            stitchedRings.push_back(res);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }

            if(debugName != "") {
                res->image.Load();
                imwrite(debugName + "_ring_" + ToString(i) + "_ev_" + ToString(ev) + ".jpg",  res->image.data); 
                res->image.Unload();

            }
            stitcherTimer.Tick("Ring Finished");
        }

        assert(margin != -1);
           
        cout << "Attempting ring adjustment." << endl;
        ringAdjustmentProgress(1);
        
        AdjustCorners(stitchedRings, ringAdjustmentProgress);
        FindSeams(stitchedRings, store);
        
        stitcherTimer.Tick("Corner Adjusting Finished");
        
	    Ptr<Blender> blender;
	    blender = Blender::createDefault(cv::detail::Blender::FEATHER, false);
        //MultiBandBlender* mb;
        //mb = dynamic_cast<MultiBandBlender*>(blender.get());
        //mb->setNumBands(5);
        

        Rect inRoi = detail::resultRoi(fun::map<StitchingResultP, Point>
                (stitchedRings, [](const StitchingResultP &x) { return x->corner; }),
                fun::map<StitchingResultP, Size>
                (stitchedRings, [](const StitchingResultP &x) { return x->image.size(); }));
        if(w == 0 && h == 0) {
            w = inRoi.width;
            h = inRoi.height;
            margin = 0;
        }
        Rect outRoi(0, 0, w, h);

        int sx = inRoi.x;
        int sy = inRoi.y - margin;
        float dx = (float)outRoi.width / (float)inRoi.width;
        float dy = (float)outRoi.height / (float)(inRoi.height + 2 * margin);

        cout << "InRoi: " << inRoi << " OutRoi: " << outRoi << endl;
        blender->prepare(outRoi);
       
        cout << "Attempting ring blending." << endl;
        for(size_t i = 0; i < stitchedRings.size(); i++) {
            finalBlendingProgress((float)i / (float)stitchedRings.size());
            auto res = stitchedRings[i];

            Mat resizedImage;
            Mat resizedMask;

            res->image.Load();
            assert(res->image.type() == CV_8UC3);
            resize(res->image.data, resizedImage, cv::Size(0, 0), dx, dy);
            res->image.Unload();

            res->mask.Load(IMREAD_GRAYSCALE);
            assert(res->mask.type() == CV_8U);
            resize(res->mask.data, resizedMask, cv::Size(0, 0), dx, dy);
            res->mask.Unload();

            Mat warpedImageAsShort;
            resizedImage.convertTo(warpedImageAsShort, CV_16S);

            //Set one pixel of the mask to black on the edges to enable blending. 
            resizedMask(Rect(0, 0, resizedMask.cols, 1)).setTo(Scalar::all(0));
            resizedMask(Rect(0, resizedMask.rows - 1, resizedMask.cols, 1)).setTo(Scalar::all(0));

            Point newCorner((res->corner.x - sx) * dx, (res->corner.y - sy) * dy); 

            blender->feed(warpedImageAsShort, resizedMask, newCorner);
        }
        
        finalBlendingProgress(1);

        //for(int i = 0; i < mb->num_bands_; i++) {
        //    Mat tmp;
        //    mb->dst_pyr_laplace_[i].convertTo(tmp, CV_8UC3, 12, 0);
        //    imwrite("dbg/dst_pyr_laplace_" + ToString(i) + ".jpg", tmp);
        //    mb->dst_band_weights_[i].convertTo(tmp, CV_8UC3, 255.0, 0);
        //    imwrite("dbg/dst_pyr_weights_" + ToString(i) + ".jpg", mb->dst_band_weights_[i]);
        //}

        stitchedRings.clear();
        {
            Mat imageRes, maskRes;
            blender->blend(imageRes, maskRes);
            imageRes.convertTo(imageRes, CV_8U);
            res->image = Image(imageRes);
            res->mask = Image(maskRes);
        }
        stitcherTimer.Tick("FinalStitching Finished");
        blender.release();
        
        store.SaveOptograph(res);
        
        stitcherTimer.Tick("Resize Finished");

        return res;
    }
}
