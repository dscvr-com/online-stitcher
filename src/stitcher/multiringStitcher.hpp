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

    /*
     * Class capable of stitching multiple rings. 
     */    
    class MultiRingStitcher {
        private:
            int w = 4096;
            int h = 4096;
            std::vector<int> dyCache; // Horizontal offsets of rings. 
            std::vector<std::vector<InputImageP>> rings; // Ring raw images.
            ExposureCompensator exposure; // Exposure compensator.
            CheckpointStore &store; // Store for storing intermediate results (saving memory).
            double ev; // Exposure bias setting.
           
            /*
             * Finds approximate horizontal offsets between the given rings. 
             */ 
            void AdjustCorners(
                    std::vector<StitchingResultP> &rings, 
                    ProgressCallback &progress);

            /*
             * Invokes stitching for a single ring.
             */
            StitchingResultP StitchRing(
                    const std::vector<InputImageP> &ring,
                    ProgressCallback &progress,
                    int ringId) const;
        public:
            /*
             * Prepares for stitching with the given width, height and store. 
             */
            MultiRingStitcher(int width, int height, CheckpointStore &store) : 
                w(width), h(height), store(store) { }

            /*
             * Prepares for stitching with default width, height and the given store. 
             */
            MultiRingStitcher(CheckpointStore &store) : 
                w(0), h(0), store(store) { }

            /*
             * Initializes the stitching engine with the given set of input images and
             * the given exposure compensator. 
             */
            void InitializeForStitching(
                    std::vector<std::vector<InputImageP>> &rings, 
                    ExposureCompensator &exposure, double ev = 0);
           
            /*
             * Starts the stitching process. 
             */ 
            StitchingResultP Stitch(
                    ProgressCallback &progress,
                    const std::string &debugName = "");

    };
}

#endif
