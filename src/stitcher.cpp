#include "image.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "monoStitcher.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
#include "ringwiseStitcher.hpp"
#include "checkpointStore.hpp"

#include "static_timer.hpp"

#include <chrono>

#ifndef OPTONAUT_STITCHER_HEADER
#define OPTONAUT_STITCHER_HEADER

namespace optonaut {
    
    class Stitcher {
        
    private:
        CheckpointStore &store;
        RingwiseStitcher core;
    public:
        
        Stitcher(CheckpointStore &store) :
        store(store), core(store) {
        }
        
        Stitcher(int width, int height, CheckpointStore &store) :
        store(store), core(width, height, store) {
        }
        
        StitchingResultP Finish(bool debug = false, string debugName = "") {
            vector<vector<InputImageP>> rings;
            ExposureCompensator exposure;
            map<size_t, double> gains;
            
            store.LoadStitcherInput(rings, gains);
            
            exposure.SetGains(gains);
            
            StitchingResultP res;
            
            core.InitializeForStitching(rings, exposure, 0.4, debug, debugName);
            res = core.Stitch(debug, debugName);
            
            return res;
        }
    };    
}

#endif
