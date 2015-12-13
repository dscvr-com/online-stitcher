#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "../common/image.hpp"
#include "../common/drawing.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"
#include "../math/stat.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "correlation.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

#ifndef OPTONAUT_PAIRWISE_CORRELATOR_HEADER
#define OPTONAUT_PAIRWISE_CORRELATOR_HEADER

namespace optonaut {

class CorrelationDiff {
public:
	bool valid;
    int overlap;
    Point2f offset;
    double horizontalAngularOffset;
    double verticalAngularOffset;
    double variance;

	CorrelationDiff() : valid(false), offset(0, 0), horizontalAngularOffset(0) {}

};

class PairwiseCorrelator {

private:
    static const bool debug = false;
    typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> Aligner;
public:
    PairwiseCorrelator() { }
   
    // Note: An outlier threshold of 2 is fine (1 pixel in each dimension), since
    // we don't do sub-pixel alignment.  
    CorrelationDiff Match(const InputImageP a, const InputImageP b, int minWidth = 0, int minHeight = 0, int outlierThreshold = 2) {

        STimer cTimer;
        CorrelationDiff result;
        
        Mat wa, wb;

        cv::Point appliedBorder;
        cv::Rect overlappingRoi = GetOverlappingRegion(a, b, a->image, b->image, wa, wb, a->image.cols * 0.2, appliedBorder);

        cTimer.Tick("Overlap found");

        if(minWidth < 4)
            minWidth = 4;

        if(minHeight < 4)
            minHeight = 4;

        //Overlap too small - invalid. 
        if(wa.cols < minWidth || wb.cols < minWidth || 
                wa.rows < minHeight || wb.rows < minHeight) {
            result.valid = false;
            return result;
        }

        PlanarCorrelationResult res = Aligner::Align(wa, wb, 0.25, 0.25, 0);
        PlanarCorrelationResult res2 = Aligner::Align(wb, wa, 0.25, 0.25, 0);

        auto diff = res.offset + res2.offset;

        //Inverse match not consistent - invalid. 
        int diffSum = diff.x * diff.x + diff.y * diff.y;
        if(diffSum > outlierThreshold) {
            //cout << "Planar Correlator Discarding: " << res.offset << " <-> " << res2.offset << " (" << diffSum << ")" << endl;
            result.valid = false;
            return result;
        }

        cv::Point correctedRes = res.offset + appliedBorder;
        
        double h = b->intrinsics.at<double>(0, 0) * (b->image.cols / (b->intrinsics.at<double>(0, 2) * 2));
        double olXA = (overlappingRoi.x + correctedRes.x - b->image.cols / 2) / h;
        double olXB = (overlappingRoi.x - b->image.cols / 2) / h;
        
        double olYA = (overlappingRoi.y + correctedRes.y - b->image.rows / 2) / h;
        double olYB = (overlappingRoi.y - b->image.rows / 2) / h;
       
        result.overlap = wa.cols * wa.rows; 
        result.horizontalAngularOffset = atan(olXA) - atan(olXB);
        result.verticalAngularOffset = atan(olYA) - atan(olYB);
        result.offset = correctedRes;
        result.valid = true;
        result.variance = res.variance;
        cTimer.Tick("Correalted");

        return result;
    }
};
}

#endif
