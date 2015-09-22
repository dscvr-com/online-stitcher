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
	public:
    StitchingResultP Stitch(std::vector<std::vector<ImageP>> &rings, bool debug = false, std::string debugName = ""); 
};
}

#endif
