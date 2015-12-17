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
    Point2f angularOffset;
    int rejectionReason;
    double variance;
    Point2f inverseTestDifference;

	CorrelationDiff() : 
        valid(false), 
        offset(0, 0), 
        angularOffset(0, 0), 
        rejectionReason(-1),
        variance(0),
        inverseTestDifference(0, 0) {}

};

class PairwiseCorrelator {

private:
    static const bool debug = false;
    typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> Aligner;
public:
    PairwiseCorrelator() { }

    static const int RejectionUnknown = -1;
    static const int RejectionNone = 0;
    static const int RejectionNoOverlap = 1;
    static const int RejectionInverseTest = 2;
   
    // Note: An outlier threshold of 2 is fine (1 pixel in each dimension), since
    // we don't do sub-pixel alignment.  
    CorrelationDiff Match(const InputImageP a, const InputImageP b, int minWidth = 0, int minHeight = 0) {

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
            result.rejectionReason = RejectionNoOverlap;
            return result;
        }

        Mat corr; //Debug stuff

        PlanarCorrelationResult res = Aligner::Align(wa, wb, corr, 0.5, 0.5, 0);

        cv::Point correctedRes = res.offset + appliedBorder;
        
        double h = b->intrinsics.at<double>(0, 0) * (b->image.cols / (b->intrinsics.at<double>(0, 2) * 2));
        double olXA = (overlappingRoi.x + correctedRes.x - b->image.cols / 2) / h;
        double olXB = (overlappingRoi.x - b->image.cols / 2) / h;
        
        double olYA = (overlappingRoi.y + correctedRes.y - b->image.rows / 2) / h;
        double olYB = (overlappingRoi.y - b->image.rows / 2) / h;
       
        result.overlap = wa.cols * wa.rows; 
        result.angularOffset.x = atan(olXA) - atan(olXB);
        result.angularOffset.y = atan(olYA) - atan(olYB);
        result.offset = correctedRes;
        result.valid = true;
        result.variance = res.variance;
        cTimer.Tick("Correalted");

        return result;
    }
};
}

#endif
