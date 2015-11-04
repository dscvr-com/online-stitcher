
#include <opencv2/core.hpp>
#include "stitchingResult.hpp"

#ifndef OPTONAUT_DP_SEAMER_HEADER
#define OPTONAUT_DP_SEAMER_HEADER

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

class DynamicSeamer 
{
private:
    static int debugId;
public:
    template <bool vertical>
    static void Find(Mat& imageA, Mat &imageB, Mat &maskA, Mat &maskB, const Point &tlA, const Point &tlB, int border, int overlap, int id);

    static inline void FindVerticalFromStitchingResult(StitchingResultP &a, StitchingResultP &b) {
        Find<true>(a->image.data, b->image.data, a->mask.data, 
             b->mask.data, a->corner, b->corner, 0, 1, debugId++);
    }

    static inline void FindHorizontalFromStitchingResult(StitchingResultP &a, StitchingResultP &b) {

        if(a->seamed && b->seamed)
             return;

        cout << "Seaming: " << a->id << ", " << b->id << endl;

        Find<false>(a->image.data, b->image.data, a->mask.data,  
             b->mask.data, a->corner, b->corner, 16, 16, debugId++);
    }
};

}
#endif
