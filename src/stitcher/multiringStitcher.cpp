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
#include "../common/drawing.hpp"
#include "../common/support.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/static_timer.hpp"
#include "../common/functional.hpp"
#include "ringStitcher.hpp"
#include "multiringStitcher.hpp"
#include "stitchingResult.hpp"
#include "dynamicSeamer.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
    
    STimer stitcherTimer;

    /*
     * Loads a stitching result from the disk cache. 
     */
    void LoadSRP(CheckpointStore&, const StitchingResultP &a) {
        if(!a->image.IsLoaded()) {
            a->image.Load();
            a->mask.Load(CV_LOAD_IMAGE_GRAYSCALE);
        }
    }
    
    /*
     * Loads a stitching result in grayscale without mask. 
     */
    void LoadSRPGrayscaleDropMask(CheckpointStore&, 
            const StitchingResultP &a) {
        if(!a->image.IsLoaded()) {
            a->image.Load(CV_LOAD_IMAGE_GRAYSCALE);
        }
    }

    /*
     * Unloads a stitching result from main memory. 
     */
    void UnLoadSRP(CheckpointStore &store, const StitchingResultP &a) {
        if(store.SupportsPaging()) {
            a->image.Unload();
            a->mask.Unload();
        }
    } 
   
    /*
     * Unloads a stitching result from main memory. Saves the image mask to the disk cache. 
     */ 
    void UnLoadSRPStoreMask(CheckpointStore &store, const StitchingResultP &a) {
        if(store.SupportsPaging()) {
            a->image.Unload();
        }
        a->seamed = true;
        cout << "Saving mask" << a->id << endl;
        if(store.SupportsPaging()) {
            store.SaveRingMask(a->id, a);
            a->mask.Unload();
        }
    }
         
    void MultiRingStitcher::AdjustCorners(std::vector<StitchingResultP> &rings, ProgressCallback &progress) {

        /*
         * Correlates an image pair using OpenCVs extended correlation coefficients. 
         * Adds the correlation result (the approximate horizontal offset) to the dy cache. 
         */
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

                if(imgA->image.data.type() != CV_8UC1) {
                    Mat grayA, grayB;

                    cvtColor(imgA->image.data, grayA, CV_BGR2GRAY);
                    cvtColor(imgB->image.data, grayB, CV_BGR2GRAY);
                    
                    findTransformECC(grayA, grayB, affine, warp, termination);
                } else {
                    findTransformECC(imgA->image.data, imgB->image.data, affine, warp, termination);
                }
                dy = affine.at<float>(1, 2);
            } catch (Exception ex) {
                // :(
            }

            dyCache.push_back(dy);
        };

        // Load ring adjustment, if we already have one (for example if we're stitching the right
        // image, and we want to use the adjustment of the left image). 
        store.LoadRingAdjustment(dyCache);

        // Setup a ring processor without overlap and process our list of rings. 
        // Process is: Load image in grayscale, correlate, then unload the image.
        if(dyCache.size() == 0) {
            RingProcessor<StitchingResultP> queue(1, 0, 
                std::bind(&LoadSRPGrayscaleDropMask, std::ref(store), std::placeholders::_1), 
                correlate, 
                std::bind(&UnLoadSRP, std::ref(store), std::placeholders::_1));
            queue.Process(rings, progress);
            
            // If we did ring adjustment, save the resulting horizontal offset to our cache. 
            store.SaveRingAdjustment(dyCache);
        }

        // Apply the horizontal offset to all rings. 
        for(size_t i = 1; i < rings.size(); i++) {
            rings[i]->corner.y = rings[i - 1]->corner.y - dyCache[i - 1];
        }
    }

    /*
     * Find seams between all rings. 
     */
    void FindSeams(std::vector<StitchingResultP> &rings, CheckpointStore &store) {

        // Setup a processing queue without overlap. 
        // Process is: Load image and mask, find seam, then unload image and mask. 
        RingProcessor<StitchingResultP> queue(1, 0, 
                std::bind(&LoadSRP, std::ref(store), std::placeholders::_1), 
                DynamicSeamer::FindHorizontalFromStitchingResult, 
                std::bind(&UnLoadSRPStoreMask, std::ref(store), std::placeholders::_1));

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
    
    StitchingResultP MultiRingStitcher::StitchRing(const vector<InputImageP> &ring, ProgressCallback &progress, int ringId) const {
        
        static const bool debug = true;

        cout << "Attempting to stitch ring " << ringId << endl;

        // Attempt to load the ring.
        StitchingResultP res = store.LoadRing(ringId);

        // If ring could be loaded, we can use it instead of stitching it again. 
        // This happens in case of continuing from a checkpoint. 
        if(res != NULL) {
            progress(1);
            return res;
        }
       
        // Otherwise, use the ring stitcher and save the result. 
        RingStitcher stitcher;
        
        res = stitcher.Stitch(ring, progress);
        res->id = ringId;

        if(debug) {
            DrawImagePointsOnPanorama(res, ring, stitcher.GetWarperScale(), Scalar(0, 0, 255));
        }

        if(store.SupportsPaging()) {
            store.SaveRing(ringId, res);

            res->image.Unload();
            res->mask.Unload();
        }

        return res;
    }

    StitchingResultP MultiRingStitcher::Stitch(ProgressCallback &progress, const string &debugName) {

        stitcherTimer.Tick("StitchStart");
       
        // Try to load the result. If we can load it, we don't have to stitch it.  
        StitchingResultP res = store.LoadOptograph();
        if(res != NULL) {
            //Caller expects a loaded image. 
            res->image.Load();
            progress(1);
            return res;
        }
        
        res = StitchingResultP(new StitchingResult());
   
        // Setup a cummulated progress callback for all stages.  
        vector<float> weights(rings.size() + 2);
        fill(weights.begin(), weights.end(), 1.0f / weights.size());
        ProgressCallbackAccumulator progressCallbacks(progress, weights);
        ProgressCallback &ringAdjustmentProgress = progressCallbacks.At(weights.size() - 2);
        ProgressCallback &finalBlendingProgress = progressCallbacks.At(weights.size() - 1);

        vector<StitchingResultP> stitchedRings;
       
        cout << "Attempting to stitch rings." << endl;
        int margin = -1; 
       
        // For each ring, stitch. 
        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) {
                progressCallbacks.At(i)(1);
                continue;
            }
            
            auto res = StitchRing(rings[i], progressCallbacks.At(i), (int)i);
            
            stitchedRings.push_back(res);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }

            if(debugName != "") {
                if(!res->image.IsLoaded()) {
                    res->image.Load();
                }
                imwrite(debugName + "_ring_" + ToString(i) + "_ev_" + 
                        ToString(ev) + ".jpg",  res->image.data); 
                if(store.SupportsPaging()) {
                    res->image.Unload();
                }

            }
            stitcherTimer.Tick("Ring Finished");
        }

        // If we have more than one ring, adjust the rings, 
        // find seams and blend them together. 
        if(stitchedRings.size() > 1) {
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

                if(!res->image.IsLoaded()) {
                    res->image.Load();
                }
                assert(res->image.type() == CV_8UC3);
                resize(res->image.data, resizedImage, cv::Size(0, 0), dx, dy);
                if(store.SupportsPaging()) {
                    res->image.Unload();
                }

                if(!res->image.IsLoaded()) {
                    res->mask.Load(IMREAD_GRAYSCALE);
                }
                assert(res->mask.type() == CV_8U);
                resize(res->mask.data, resizedMask, cv::Size(0, 0), dx, dy);
                if(store.SupportsPaging()) {
                    res->mask.Unload();
                }

                Mat warpedImageAsShort;
                resizedImage.convertTo(warpedImageAsShort, CV_16S);

                //Set one pixel of the mask to black on the edges to enable blending. 
                resizedMask(Rect(0, 0, resizedMask.cols, 1)).setTo(Scalar::all(0));
                resizedMask(Rect(0, resizedMask.rows - 1, resizedMask.cols, 1)).setTo(Scalar::all(0));

                Point newCorner((res->corner.x - sx) * dx, (res->corner.y - sy) * dy); 

                blender->feed(warpedImageAsShort, resizedMask, newCorner);
            }

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
            }
            stitcherTimer.Tick("FinalStitching Finished");
            blender.release();
        } else {
            res = stitchedRings.front();
            if(!res->image.IsLoaded()) {
                res->image.Load();
            }
        }
        
        
        // Final Step, trim away a few pixels to avoid masking issues.
        static const int trim = 2;
        
        res->image = Image(res->image.data(cv::Rect(0, trim, res->image.cols, res->image.rows - trim * 2)));
        
        store.SaveOptograph(res);
        
        finalBlendingProgress(1);
        
        stitcherTimer.Tick("Resize Finished");

        return res;
    }
}
