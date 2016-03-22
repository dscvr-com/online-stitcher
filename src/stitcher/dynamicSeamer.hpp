
#include <opencv2/core.hpp>
#include "stitchingResult.hpp"

#ifndef OPTONAUT_DP_SEAMER_HEADER
#define OPTONAUT_DP_SEAMER_HEADER

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

/*
 * Finds min-cost cuts between two images, either horizontally or vertically. 
 */
class DynamicSeamer 
{
private:
    static int debugId;
public:
    /*
     * Finds a min-cost cut between imageA and imabeB and updates maskA and maskB accordingly.
     *
     * @param vertical Indicates wether the cut should be done vertically or horizontally. 
     * @param imageA The first image.
     * @param imageB The second image.
     * @param maskA The mask of the first image. 
     * @param maskB The mask of the second image. 
     * @param tlA The top left corner of the first image, in global coordinate space. 
     * @param tlB The top left corner of the second image, in global coordinate space. 
     * @param border Margin to keen unused between the image border and the cut. 
     * @param overlap Thickness of the seam area, that is the margin that is added to the left and the right of the seam, so the images overlap. 
     * @param id Image id. Used for debugging. 
     */
    template <bool vertical>
    static void Find(Mat& imageA, Mat &imageB, Mat &maskA, Mat &maskB, const Point &tlA, const Point &tlB, int border, int overlap, int id);

    /*
     * Invokes seam finding for a pair of stitching results. 
     */
    static inline void FindVerticalFromStitchingResult(const StitchingResultP &a, const StitchingResultP &b) {
        Find<true>(a->image.data, b->image.data, a->mask.data, 
             b->mask.data, a->corner, b->corner, 0, 1, debugId++);
    }

    /*
     * Invokes seam finding for a pair of stitching results. 
     */
    static inline void FindHorizontalFromStitchingResult(const StitchingResultP &a, const StitchingResultP &b) {

        if(a->seamed && b->seamed)
             return;

        cout << "Seaming: " << a->id << ", " << b->id << endl;

        Find<false>(a->image.data, b->image.data, a->mask.data,  
             b->mask.data, a->corner, b->corner, 16, 32, debugId++);
    }
};

}
#endif
