#include <opencv2/opencv.hpp>
#include <vector>

#include "../common/image.hpp"

#ifndef OPTONAUT_STITCHING_RESULT_HEADER
#define OPTONAUT_STITCHING_RESULT_HEADER

namespace optonaut {
    /*
     * Represents a stitched image.
     */
    struct StitchingResult {
        Image image; // The stitching result. 
        Image mask; // The alpha channel 
        cv::Point corner; // The top-left corner of the most top-left image that was stitched into the result.
        int id; // The id of the result
        bool seamed; // A boolean indicating wether the result image was seamed already. 
    };

    typedef std::shared_ptr<StitchingResult> StitchingResultP;
}

#endif
