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
    StitchingResultP RingwiseStitcher::Stitch(vector<vector<ImageP>> &rings, bool debug, string debugName) {

        RStitcher stitcher;

        vector<StitchingResultP> stitchedRings(rings.size());
        vector<cv::Size> sizes(rings.size());
        vector<cv::Point> corners(rings.size());
        
        Ptr<Blender> blender = Blender::createDefault(Blender::FEATHER, true);

        for(size_t i = 0; i < rings.size(); i++) {
            if(rings[i].size() == 0) 
                continue;

            auto res = stitcher.Stitch(rings[i], debug);
            stitchedRings[i] = res;
            sizes[i] = res->image.size();
            corners[i] = res->corner;
        }
            
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

        return res;
    }
}
