#include "../common/image.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
#include "../io/checkpointStore.hpp"

#include "multiRingStitcher.hpp"

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

        StitchingResultP Finish(ProgressCallback &progress, bool debug = false, string debugName = "") {
            vector<vector<InputImageP>> rings;
            ExposureCompensator exposure;
            map<size_t, double> gains;

            store.LoadStitcherInput(rings, gains);
            
            exposure.SetGains(gains);
            
            StitchingResultP res;
            
            core.InitializeForStitching(rings, exposure, 0.4);
            res = core.Stitch(progress, debug, debugName);

            return res;
        }
    };    
}

#endif
