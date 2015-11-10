#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "../math/support.hpp"
#include "../stitcher/stitchingResult.hpp"
#include "../io/inputImage.hpp"

namespace optonaut {
//Fast pure R-Matrix based stitcher
class SimpleSphereStitcher {
	public:
		float workScale = 0.2f;
		float warperScale = 800;

		StitchingResultP Stitch(const std::vector<InputImageP> &images, bool debug = false) const;
};
}
