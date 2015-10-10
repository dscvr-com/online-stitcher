#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "ringwiseStitcher.hpp"
#include "support.hpp"
#include "correlation.hpp"
#include "static_timer.hpp"
#include "stitchingResult.hpp"


using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

    void RingwiseStitcher::AdjustCorners(std::vector<StitchingResultP> &rings, std::vector<cv::Point> &corners, ProgressCallback &progress) {
        
        if(dyCache.size() == 0) {
            rings[0]->image.Load();
            Mat a = rings[0]->image.data;
            for(size_t i = 1; i < rings.size(); i++) {
                progress((float)i / (float)rings.size());
                
                rings[i]->image.Load();
                Mat b = rings[i]->image.data;
                Mat ca, cb;
                
                cvtColor(a, ca, CV_BGR2GRAY);
                cvtColor(b, cb, CV_BGR2GRAY); 

                const int warp = MOTION_TRANSLATION;
                Mat affine = Mat::eye(2, 3, CV_32F);

                const int iterations = 100;
                const double eps = 1e-3;

                int dy = corners[i - 1].y - corners[i].y;
                affine.at<float>(1, 2) = dy;

                TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
                try {
                    findTransformECC(ca, cb, affine, warp, termination);
                    dy = affine.at<float>(1, 2);
                } catch (Exception ex) {
                    // :( 
                }

                dyCache.push_back(dy);
                
                a = b;
                rings[i - 1]->image.Unload();
            }
            rings.back()->image.Unload();
        }
        
        progress(1);

        for(size_t i = 1; i < rings.size(); i++) {
            corners[i].y = corners[i - 1].y - dyCache[i - 1];
            rings[i]->corner = corners[i];
        }
    }
    
    void RingwiseStitcher::InitializeForStitching(std::vector<std::vector<InputImageP>> &rings, ExposureCompensator &exposure, double ev) {
        
        this->rings = rings;
        this->exposure.SetGains(exposure.GetGains());
        this->ev = ev;
        this->dyCache = vector<int>();
    }
    
    StitchingResultP RingwiseStitcher::StitchRing(const vector<InputImageP> &ring, ProgressCallback &progress, int ringId, bool debug, const string &debugName) const {
        
        cout << "Attempting to stitch ring " << ringId << endl;

        StitchingResultP res = store.LoadRing(ringId);

        if(res != NULL) {
            progress(1);
            return res;
        }
        
        RStitcher stitcher(store);
        
        res = stitcher.Stitch(ring, exposure, progress, ev, debug, debugName);

        store.SaveRing(ringId, res);

        res->image.Unload();
        res->mask.Unload();

        return res;
    }

    StitchingResultP RingwiseStitcher::Stitch(ProgressCallback &progress, bool debug, const string &debugName) {

        STimer::Tick("StitchStart");
        
        StitchingResultP res = store.LoadOptograph();
        if(res != NULL) {
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
        vector<cv::Size> sizes;
        vector<cv::Point> corners;
       
        FeatherBlender* pblender = new FeatherBlender(0.01f); 
        Ptr<Blender> blender = Ptr<Blender>(pblender);

        cout << "Attempting to stitch rings." << endl;
        int margin = -1; 
        
        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) {
                progressCallbacks.At(i)(1);
                continue;
            }
            
            auto res = StitchRing(rings[i], progressCallbacks.At(i), i, debug, debugName);
            
            stitchedRings.push_back(res);
            sizes.push_back(res->image.size());
            corners.push_back(res->corner);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }

            if(debugName != "") {
                res->image.Load();
                imwrite(debugName + "_ring_" + ToString(i) + "_ev_" + ToString(ev) + ".jpg",  res->image.data); 
                res->image.Unload();

            }
            STimer::Tick("Ring Finished");
        }

        assert(margin != -1);
           
        cout << "Attempting ring adjustment." << endl;
        ringAdjustmentProgress(1); 
        //AdjustCorners(stitchedRings, corners, ringAdjustmentProgress);
        
        STimer::Tick("Corner Adjusting Finished");
        blender->prepare(corners, sizes);
        
        cout << "Attempting ring blending." << endl;
        for(size_t i = 0; i < stitchedRings.size(); i++) {
            finalBlendingProgress((float)i / (float)stitchedRings.size());
            auto res = stitchedRings[i];
            res->image.Load();
            res->mask.Load(IMREAD_GRAYSCALE);
            
            assert(res->image.type() == CV_8UC3);
            assert(res->mask.type() == CV_8U);

            Mat warpedImageAsShort;
            res->image.data.convertTo(warpedImageAsShort, CV_16S);

            Mat mask = res->mask.data;

            //Set one pixel of the mask to black on the edges to enable blending. 
            mask(Rect(0, 0, mask.cols, 1)).setTo(Scalar::all(0));
            mask(Rect(0, mask.rows - 1, mask.cols, 1)).setTo(Scalar::all(0));

            blender->feed(warpedImageAsShort, mask, corners[i]);

            res->image.Unload();
            res->mask.Unload();
        }
        
        finalBlendingProgress(1);

        stitchedRings.clear();
        {
            Mat imageRes, maskRes;
            blender->blend(imageRes, maskRes);
            imageRes.convertTo(imageRes, CV_8U);
            res->image = Image(imageRes);
            res->mask = Image(maskRes);
        }
        STimer::Tick("FinalStitching Finished");
        blender.release();
        
        //Opencv somehow messes up the first few collumn while blending.
        //Throw it away. 
        const int trim = 6;
        res->image = Image(res->image.data(cv::Rect(trim, 0, res->image.cols - trim * 2, res->image.rows)));
        res->mask = Image(res->mask.data(cv::Rect(trim, 0, res->mask.cols - trim * 2, res->mask.rows)));

        if(resizeOutput) {

            int ih = (res->image.rows) * h / (res->image.rows + 2 * margin);
            int x = (h - ih) / 2;

            static const bool needMask = false;
            
            {
                Mat canvas(w, h, CV_8UC3);
                canvas.setTo(Scalar::all(0));
                Mat resizedImage(w, ih, CV_8UC3);
                resize(res->image.data, resizedImage, cv::Size(w, ih));
                resizedImage.copyTo(canvas.rowRange(x, x + ih));
                res->image = Image(canvas);
            }
            if(needMask) {
                Mat maskCanvas(w, h, CV_8U);
                maskCanvas.setTo(Scalar::all(0));
                Mat resizedMask(w, ih, CV_8U);
                resize(res->mask.data, resizedMask, cv::Size(w, ih));
                resizedMask.copyTo(maskCanvas.rowRange(x, x + ih));
                res->mask = Image(maskCanvas);
            } else {
                res->mask = Image(Mat(0, 0, CV_8UC1));
            }
        }
        
        store.SaveOptograph(res);
        
        STimer::Tick("Resize Finished");

        return res;
    }
}
