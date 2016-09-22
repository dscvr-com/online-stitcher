#include <vector>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/opencv.hpp>

#include "../math/support.hpp"
#include "../stitcher/stitchingResult.hpp"
#include "../io/inputImage.hpp"

#ifndef OPTONAUT_SIMPLE_SPHERE_STITCHER_HEADER
#define OPTONAUT_SIMPLE_SPHERE_STITCHER_HEADER

namespace optonaut {
/*
 * Fast rotational model stitcher. No seaming, no blending. 
 */
class SimpleSphereStitcher {
    private: 
        cv::detail::SphericalWarper warper; //Random warper scale. 
	public:

        SimpleSphereStitcher(float warperScale = 800) : warper(warperScale) {
            
        }

		StitchingResultP Stitch(const std::vector<InputImageP> &images, bool smallImages = false, bool drawRotationCenters = false);
        cv::Rect Warp(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const cv::Size &imageSize);
       
        // Point is relative to image center.  
        cv::Point WarpPoint(const cv::Mat &intrinsics, const cv::Mat &extrinsics, const cv::Size &imageSize, const cv::Point &point);
        inline cv::detail::RotationWarper& GetWarper() { return warper; }

        static inline StitchingResultP StitchAndWrite(const std::string &path, const std::vector<InputImageP> &images) {
            SimpleSphereStitcher s;
            auto res = s.Stitch(images);
            cv::imwrite(path, res->image.data);
            return res;
        }
};
}

#endif
