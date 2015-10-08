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


using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

    void RingwiseStitcher::AdjustCorners(std::vector<StitchingResultP> &rings, std::vector<cv::Point> &corners) {
        //dyCache. 
        if(dyCache.size() == 0) {
            for(size_t i = 1; i < rings.size(); i++) {
                Mat &a = rings[i - 1]->image;
                Mat &b = rings[i]->image;
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

                //cout << "V-Correlate: " << dy << endl;
            }
        }

        for(size_t i = 1; i < rings.size(); i++) {
            corners[i].y = corners[i - 1].y - dyCache[i - 1];
            rings[i]->corner = corners[i];
        }
    }
    
    void RingwiseStitcher::InitializeForStitching(std::vector<std::vector<ImageP>> &rings, ExposureCompensator &exposure, double ev) {
        
        assert(!HasCheckpoint());
        
        this->rings = rings;
        this->exposure.SetGains(exposure.GetGains());
        this->ev = ev;
        this->dyCache = vector<int>();
        
        Checkpoint();
    }
    
    bool RingwiseStitcher::HasCheckpoint() {
        return false;
    }
    
    void RingwiseStitcher::Checkpoint() {
        //TODO Save rings and adjusted corners.
    }
    
    void RingwiseStitcher::InitializeFromCheckpoint(){
        assert(HasCheckpoint());
    }
    
    void RingwiseStitcher::RemoveCheckpoint() {
        assert(HasCheckpoint());
    }
    
    StitchingResultP Stitch(bool debug = false, std::string debugName = "");
    
    StitchingResultP RingwiseStitcher::StitchRing(const vector<ImageP> &ring, bool debug, const string &debugName) {
        
        //TODO: Do not stitch if there is a file available
        RStitcher stitcher(store);
        
        return stitcher.Stitch(ring, exposure, ev, debug);
    }

    StitchingResultP RingwiseStitcher::Stitch(bool debug, const string &debugName) {

        STimer::Tick("StitchStart");

        vector<StitchingResultP> stitchedRings;
        vector<cv::Size> sizes;
        vector<cv::Point> corners;
       
        FeatherBlender* pblender = new FeatherBlender(0.01f); 
        Ptr<Blender> blender = Ptr<Blender>(pblender);

        int margin = -1; //Size_t max
        
        //TODO: find a way to recreate ring map

        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) 
                continue;
            
            auto res = StitchRing(rings[i], debug, debugName);
            
            stitchedRings.push_back(res);
            sizes.push_back(res->image.size());
            corners.push_back(res->corner);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }

            if(debugName != "") {

                imwrite("dbg/ring_" + debugName + ToString(i) + "_ev_" + ToString(ev) + ".jpg",  res->image); 

            }
            STimer::Tick("Ring Finished");
        }

        assert(margin != -1);
            
        AdjustCorners(stitchedRings, corners);
        
        STimer::Tick("Corner Adjusting Finished");
        blender->prepare(corners, sizes);

        for(size_t i = 0; i < stitchedRings.size(); i++) {
            auto res = stitchedRings[i];
            Mat warpedImageAsShort;
            res->image.convertTo(warpedImageAsShort, CV_16S);
            assert(res->mask.type() == CV_8U);

            res->mask(Rect(0, 0, res->mask.cols, 1)).setTo(Scalar::all(0));
            res->mask(Rect(0, res->mask.rows - 1, res->mask.cols, 1)).setTo(Scalar::all(0));

            blender->feed(warpedImageAsShort, res->mask, corners[i]);
        }

        stitchedRings.clear();
        STimer::Tick("FinalStitching Finished");

        StitchingResultP res(new StitchingResult());
        blender->blend(res->image, res->mask);

        blender.release();
        
        res->image.convertTo(res->image, CV_8U);
        //Opencv somehow messes up the first few collumn while blending.
        //Throw it away. 
        const int trim = 6;
        res->image = res->image(cv::Rect(trim, 0, res->image.cols - trim * 2, res->image.rows));
        res->mask = res->mask(cv::Rect(trim, 0, res->mask.cols - trim * 2, res->mask.rows));

        if(resizeOutput) {

            int ih = (res->image.rows) * h / (res->image.rows + 2 * margin);
            int x = (h - ih) / 2;

            static const bool needMask = false;
            
            {
                Mat canvas(w, h, CV_8UC3);
                canvas.setTo(Scalar::all(0));
                Mat resizedImage(w, ih, CV_8UC3);
                resize(res->image, resizedImage, cv::Size(w, ih));
                resizedImage.copyTo(canvas.rowRange(x, x + ih));
                res->image = canvas;
            }
            if(needMask) {
                Mat maskCanvas(w, h, CV_8U);
                maskCanvas.setTo(Scalar::all(0));
                Mat resizedMask(w, ih, CV_8U);
                resize(res->mask, resizedMask, cv::Size(w, ih));
                resizedMask.copyTo(maskCanvas.rowRange(x, x + ih));
                res->mask = maskCanvas;
            } else {
                res->mask = Mat(0, 0, CV_8UC1);
            }
        }
        STimer::Tick("Resize Finished");

        return res;
    }
}
