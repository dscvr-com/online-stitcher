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
    Point2f offset;
    double horizontalAngularOffset;
    double variance;

	CorrelationDiff() : valid(false), offset(0, 0), horizontalAngularOffset(0) {}

};

class PairwiseCorrelator {

private:
    static const bool debug = false;
    typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> Aligner;
public:
    PairwiseCorrelator() { }

    CorrelationDiff Match(const InputImageP a, const InputImageP b) {

        STimer cTimer;
        CorrelationDiff result;
        
        Mat wa, wb;

        cv::Point appliedBorder;
        cv::Rect overlappingRoi = GetOverlappingRegion(a, b, a->image, b->image, wa, wb, a->image.cols * 0.2, appliedBorder);

        cTimer.Tick("Overlap found");

        PlanarCorrelationResult res = Aligner::Align(wa, wb, 0.25, 0.25, 0);
        cv::Point correctedRes = res.offset + appliedBorder;
        
        double h = b->intrinsics.at<double>(0, 0) * (b->image.cols / (b->intrinsics.at<double>(0, 2) * 2));
        double olXA = (overlappingRoi.x + correctedRes.x - b->image.cols / 2) / h;
        double olXB = (overlappingRoi.x - b->image.cols / 2) / h;
        
        result.horizontalAngularOffset = sin(olXA) - sin(olXB);
        result.offset = correctedRes;
        result.valid = true;
        result.variance = res.variance;
        cTimer.Tick("Correalted");

        return result;
    }
};
}

#endif
