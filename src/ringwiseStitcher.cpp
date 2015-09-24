#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "ringwiseStitcher.hpp"
#include "support.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

    static const bool resizeOutput = false;
    static const int w = 4096;
    static const int h = 4096;

    StitchingResultP RingwiseStitcher::Stitch(vector<vector<ImageP>> &rings, bool debug, string debugName) {

        RStitcher stitcher;

        vector<StitchingResultP> stitchedRings(rings.size());
        vector<cv::Size> sizes;
        vector<cv::Point> corners;
        
        Ptr<Blender> blender = Blender::createDefault(Blender::FEATHER, true);

        int margin = -1; //Size_t max

        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) 
                continue;

            auto res = stitcher.Stitch(rings[i], debug);
            stitchedRings[i] = res;
            sizes.push_back(res->image.size());
            corners.push_back(res->corner);

            if(margin == -1 || margin > res->corner.y) {
                margin = res->corner.y;
            }
        }

        assert(margin != -1);
            
        blender->prepare(corners, sizes);

        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) 
                continue;

            auto res = stitchedRings[i];
            Mat warpedImageAsShort;
            res->image.convertTo(warpedImageAsShort, CV_16S);
            assert(res->mask.type() == CV_8U);
            blender->feed(warpedImageAsShort, res->mask, res->corner);

            if(debugName != "") {
                imwrite("dbg/ring_" + debugName + ToString(i) + ".jpg",  res->image); 
            }
        }

        StitchingResultP res(new StitchingResult());
        blender->blend(res->image, res->mask);

        blender.release();
        
        res->image.convertTo(res->image, CV_8U);

        if(resizeOutput) {
            Mat canvas(w, h, CV_8UC3);
            canvas.setTo(Scalar::all(0));

            int ih = (res->image.rows) * h / (res->image.rows + 2 * margin);

            Mat resizedImage(w, ih, CV_8UC3);
            resize(res->image, resizedImage, cv::Size(w, ih));
            int x = (h - ih) / 2;

            resizedImage.copyTo(canvas.rowRange(x, x + ih));

            //TODO - Also copy mask
            res->image = canvas;
            res->mask.release();
        }

        return res;
    }
}
