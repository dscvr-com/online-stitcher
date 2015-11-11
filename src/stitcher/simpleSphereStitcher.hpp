#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/opencv.hpp>

#include "../math/support.hpp"
#include "../stitcher/stitchingResult.hpp"
#include "../io/inputImage.hpp"

namespace optonaut {
//Fast pure R-Matrix based stitcher
class SimpleSphereStitcher {
    private: 
        cv::detail::SphericalWarper warper; //Random warper scale. 
	public:

        SimpleSphereStitcher() : warper(800) {
            
        }

		StitchingResultP Stitch(const std::vector<InputImageP> &images, bool debug = false);
        cv::Point2f Warp(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const cv::Size &imageSize);
};
}
