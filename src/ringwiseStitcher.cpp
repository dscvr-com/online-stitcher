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

                const int iterations = 1000;
                const double eps = 1e-5;

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

                cout << "V-Correlate: " << dy << endl;
            }
        }

        for(size_t i = 1; i < rings.size(); i++) {
            corners[i].y = corners[i - 1].y - dyCache[i - 1];
            rings[i]->corner = corners[i];
        }
    }

    StitchingResultP RingwiseStitcher::Stitch(vector<vector<ImageP>> &rings, bool debug, string debugName) {

        RStitcher stitcher;

        vector<StitchingResultP> stitchedRings;
        vector<cv::Size> sizes;
        vector<cv::Point> corners;
        
        Ptr<Blender> blender = Blender::createDefault(Blender::FEATHER, true);

        int margin = -1; //Size_t max

        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) 
                continue;

            auto res = stitcher.Stitch(rings[i], debug);
            stitchedRings.push_back(res);
            sizes.push_back(res->image.size());
            corners.push_back(res->corner);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }
            
            if(debugName != "") {
                imwrite("dbg/ring_" + debugName + ToString(i) + ".jpg",  res->image); 
            }
        }

        assert(margin != -1);
            
        AdjustCorners(stitchedRings, corners);
        
        blender->prepare(corners, sizes);

        for(size_t i = 0; i < stitchedRings.size(); i++) {
            auto res = stitchedRings[i];
            Mat warpedImageAsShort;
            res->image.convertTo(warpedImageAsShort, CV_16S);
            assert(res->mask.type() == CV_8U);
            blender->feed(warpedImageAsShort, res->mask, corners[i]);
        }

        StitchingResultP res(new StitchingResult());
        blender->blend(res->image, res->mask);

        blender.release();
        
        res->image.convertTo(res->image, CV_8U);

        if(resizeOutput) {
            Mat canvas(w, h, CV_8UC3);
            Mat maskCanvas(w, h, CV_8U);
            canvas.setTo(Scalar::all(0));

            int ih = (res->image.rows) * h / (res->image.rows + 2 * margin);

            Mat resizedImage(w, ih, CV_8UC3);
            Mat resizedMask(w, ih, CV_8U);
            resize(res->image, resizedImage, cv::Size(w, ih));
            resize(res->mask, resizedMask, cv::Size(w, ih));
            int x = (h - ih) / 2;

            resizedImage.copyTo(canvas.rowRange(x, x + ih));
            resizedMask.copyTo(maskCanvas.rowRange(x, x + ih));

            //TODO - Also copy mask
            res->image = canvas;
            res->mask = maskCanvas;
        }

        return res;
    }
}
