#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "image.hpp"
#include "exposureCompensator.hpp"
#include "support.hpp"
#include "checkpointStore.hpp"
#include "progressCallback.hpp"

#ifndef OPTONAUT_RSTITCHER_HEADER
#define OPTONAUT_RSTITCHER_HEADER

namespace optonaut {

//Fast pure R-Matrix based stitcher
class RStitcher {
	public:
		bool compensate = false;
		float workScale = 0.2f;
		float warperScale = 800;
        int blendMode = -1;
        CheckpointStore &store;

        RStitcher(CheckpointStore &store) : store(store) { }

    StitchingResultP Stitch(const std::vector<InputImageP> &images, ExposureCompensator &exposure, ProgressCallback &progress, double ev = 0, bool debug = false, const std::string &debugName = "");
		static void PrepareMatrices(const std::vector<InputImageP> &r);
};
}

#endif
