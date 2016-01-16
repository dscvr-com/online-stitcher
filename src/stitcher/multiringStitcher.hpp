#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../common/image.hpp"
#include "../common/progressCallback.hpp"
#include "../math/support.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "../io/checkpointStore.hpp"

#include "ringStitcher.hpp"
#include "stitchingResult.hpp"

#ifndef OPTONAUT_RINGWISE_STITCHER_HEADER
#define OPTONAUT_RINGWISE_STITCHER_HEADER

namespace optonaut {
    
    class MultiRingStitcher {
        private:
            int w = 4096;
            int h = 4096;
            std::vector<int> dyCache;
            std::vector<std::vector<InputImageP>> rings;
            ExposureCompensator exposure;
            CheckpointStore &store;
            double ev;
            
            void AdjustCorners(
                    std::vector<StitchingResultP> &rings, 
                    ProgressCallback &progress);

            void Checkpoint();
            StitchingResultP StitchRing(
                    const std::vector<InputImageP> &ring,
                    ProgressCallback &progress,
                    int ringId) const;
        public:
            MultiRingStitcher(int width, int height, CheckpointStore &store) : 
                w(width), h(height), store(store) { }

            MultiRingStitcher(CheckpointStore &store) : 
                w(0), h(0), store(store) { }

            void InitializeForStitching(
                    std::vector<std::vector<InputImageP>> &rings, 
                    ExposureCompensator &exposure, double ev = 0);
            
            StitchingResultP Stitch(
                    ProgressCallback &progress,
                    const std::string &debugName = "");

    };
}

#endif
