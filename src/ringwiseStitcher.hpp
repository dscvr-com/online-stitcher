#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "exposureCompensator.hpp"
#include "checkpointStore.hpp"
#include "stitchingResult.hpp"
#include "progressCallback.hpp"

#ifndef OPTONAUT_RINGWISE_STITCHER_HEADER
#define OPTONAUT_RINGWISE_STITCHER_HEADER

namespace optonaut {
    
    class RingwiseStitcher {
        private:
            bool resizeOutput = true;
            int w = 4096;
            int h = 4096;
            std::vector<int> dyCache;
            std::vector<std::vector<InputImageP>> rings;
            ExposureCompensator exposure;
            CheckpointStore &store;
            double ev;
            
            void AdjustCorners(
                    std::vector<StitchingResultP> &rings, 
                    std::vector<cv::Point> &corners,
                    ProgressCallback &progress);

            void Checkpoint();
            StitchingResultP StitchRing(
                    const std::vector<InputImageP> &ring,
                    ProgressCallback &progress,
                    bool debug, 
                    const std::string &debugName);
        public:
            RingwiseStitcher(int width, int height, CheckpointStore &store) : 
                resizeOutput(true), w(width), h(height), store(store) { }

            RingwiseStitcher(CheckpointStore &store) : 
                resizeOutput(false), w(0), h(0), store(store) { }

            void InitializeForStitching(
                    std::vector<std::vector<InputImageP>> &rings, 
                    ExposureCompensator &exposure, double ev = 0);
            
            StitchingResultP Stitch(
                    ProgressCallback &progress,
                    bool debug = false,
                    const std::string &debugName = "");

    };
}

#endif
