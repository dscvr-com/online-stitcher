
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
    static void Find(Mat& imageA, Mat &imageB, Mat &maskA, Mat &maskB, const Point &tlA, const Point &tlB, bool vertical = true, int overlap = 0, int id = 0);

    static inline void FindVerticalFromStitchingResult(StitchingResultP &a, StitchingResultP &b) {
        Find(a->image.data, b->image.data, a->mask.data, 
             b->mask.data, a->corner, b->corner, true, 1, debugId++);
    }

    static inline void FindHorizontalFromStitchingResult(StitchingResultP &a, StitchingResultP &b) {

        cout << "Seaming: " << a->id << ", " << b->id << endl;

        Find(a->image.data, b->image.data, a->mask.data,  
             b->mask.data, a->corner, b->corner, false, 1, debugId++);
    }
};

}
#endif
