#include <opencv2/opencv.hpp>
#include <vector>

#include "../common/image.hpp"

#ifndef OPTONAUT_STITCHING_RESULT_HEADER
#define OPTONAUT_STITCHING_RESULT_HEADER

namespace optonaut {
    struct StitchingResult {
        Image image;
        Image mask;
        //Most top-right corner.
        cv::Point corner;
        int id;
        bool seamed;
    };

    typedef std::shared_ptr<StitchingResult> StitchingResultP;
}

#endif
