#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "image.hpp"
#include "support.hpp"
#include "simpleSphereStitcher.hpp"

#ifndef OPTONAUT_RINGWISE_STITCHER_HEADER
#define OPTONAUT_RINGWISE_STITCHER_HEADER

namespace optonaut {
    class RingwiseStitcher {
        private:
            bool resizeOutput = true;
            int w = 4096;
            int h = 4096;
            std::vector<int> dyCache; 
        public:
            RingwiseStitcher(int width, int height) : resizeOutput(true), w(width), h(height) { }
            RingwiseStitcher() : resizeOutput(false) {  }

            StitchingResultP Stitch(std::vector<std::vector<ImageP>> &rings, bool debug = false, std::string debugName = ""); 

            void AdjustCorners(std::vector<StitchingResultP> &rings, std::vector<cv::Point> &corners);
    };
}

#endif
