#include <opencv2/opencv.hpp>
#include <vector>
#include "image.hpp"

#ifndef OPTONAUT_STITCHING_RESULT_HEADER
#define OPTONAUT_STITCHING_RESULT_HEADER

namespace optonaut {
    struct StitchingResult {
        Image image;
        Image mask;
        std::vector<cv::Point> corners;
        std::vector<cv::Size> sizes;
        //Most top-right corner.
        cv::Point corner;
    };

    typedef std::shared_ptr<StitchingResult> StitchingResultP;
}

#endif
