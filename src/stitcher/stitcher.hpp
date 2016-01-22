#include "../common/image.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
#include "../io/checkpointStore.hpp"

#include "multiringStitcher.hpp"

#include <chrono>

#ifndef OPTONAUT_STITCHER_HEADER
#define OPTONAUT_STITCHER_HEADER

namespace optonaut {
    
    class Stitcher {

    private:    
        CheckpointStore &store;
        MultiRingStitcher core;
    public:

        Stitcher(CheckpointStore &store) :
            store(store), core(store) {
        }
        
        Stitcher(int width, int height, CheckpointStore &store) :
            store(store), core(width, height, store) {
        }

        StitchingResultP Finish(ProgressCallback &progress, std::string debugName = "") {
            vector<vector<InputImageP>> rings;
            ExposureCompensator exposure;
            map<size_t, double> gains;

            store.LoadStitcherInput(rings, gains);
            
            exposure.SetGains(gains);
            
            StitchingResultP res;
            
            core.InitializeForStitching(rings, exposure, 0.4);
            res = core.Stitch(progress, debugName);

            //Mat intrinsics = rings[0][0]->intrinsics;
            //cv::Size size = cv::Size(intrinsics.at<double>(0, 2),
            //        intrinsics.at<double>(1, 2));
            
            //DrawPointsOnPanorama(res->image.data,
            //        ExtractExtrinsics(fun::flat(rings)),
            //        intrinsics, size, 1200,
            //        res->corner + cv::Point(0, 10), Scalar(0xFF, 0x00, 0x00));


            return res;
        }
    };    
}

#endif
