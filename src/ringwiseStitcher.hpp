#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"
#include "exposureCompensator.hpp"
#include "checkpointStore.hpp"

#ifndef OPTONAUT_RINGWISE_STITCHER_HEADER
#define OPTONAUT_RINGWISE_STITCHER_HEADER

namespace optonaut {
    class RingwiseStitcher {
        private:
            bool resizeOutput = true;
            int w = 4096;
            int h = 4096;
            std::vector<int> dyCache;
            std::vector<std::vector<ImageP>> rings;
            ExposureCompensator exposure;
            CheckpointStore &store;
            double ev;
            
            void AdjustCorners(std::vector<StitchingResultP> &rings, std::vector<cv::Point> &corners);
            void Checkpoint();
        public:
            RingwiseStitcher(int width, int height, CheckpointStore &store) : resizeOutput(true), w(width), h(height), store(store) { }
            RingwiseStitcher(CheckpointStore &store) : resizeOutput(false), store(store) {  }
        
            void InitializeForStitching(std::vector<std::vector<ImageP>> &rings, ExposureCompensator &exposure, double ev = 0);
            bool HasCheckpoint();
            void InitializeFromCheckpoint();
            void RemoveCheckpoint();
            StitchingResultP Stitch(bool debug = false, std::string debugName = "");

    };
}

#endif
