#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "../math/support.hpp"
#include "../stitcher/stitchingResult.hpp"
#include "../common/image.hpp"

#ifndef OPTONAUT_PLANE_SPHERE_STITCHER_HEADER
#define OPTONAUT_PLANE_SPHERE_STITCHER_HEADER

namespace optonaut {
//Fast pure Planar based stitcher
class SimplePlaneStitcher {
	public:
		StitchingResultP Stitch(const std::vector<ImageP> &images, const std::vector<cv::Point> &corners) const;
};
}
#endif
