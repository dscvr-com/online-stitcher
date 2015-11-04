#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../common/image.hpp"
#include "../common/progressCallback.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "../math/support.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_RSTITCHER_HEADER
#define OPTONAUT_RSTITCHER_HEADER

namespace optonaut {

//Fast pure R-Matrix based stitcher
class RingStitcher {
	public:
		bool compensate = false;
		float workScale = 0.2f;
		float warperScale = 1200;
        CheckpointStore &store;

        RingStitcher(CheckpointStore &store) : store(store) { }

    StitchingResultP Stitch(const std::vector<InputImageP> &images, const ExposureCompensator &exposure, ProgressCallback &progress, double ev = 0, bool debug = false, const std::string &debugName = "");
		static void PrepareMatrices(const std::vector<InputImageP> &r);
};
}

#endif
