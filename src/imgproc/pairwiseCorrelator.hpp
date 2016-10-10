#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "../common/image.hpp"
#include "../common/static_timer.hpp"
#include "../common/drawing.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"
#include "../math/stat.hpp"
#include "../recorder/exposureCompensator.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

#ifndef OPTONAUT_PAIRWISE_CORRELATOR_HEADER
#define OPTONAUT_PAIRWISE_CORRELATOR_HEADER

namespace optonaut {

/*
 * Represents a position correspondence between two images,
 * based on correlation based alignment.  
 */
class CorrelationDiff {
public:
    /*
     * True, if this diff is valid, false if the diff is an outlier.  
     */
	bool valid;
    /*
     * The overlap between the correlated images, in pixels. 
     */
    int overlap;
    /*
     * The corrected offset between the correlated images, in pixels, not including any
     * difference reflected in the image's intrinsics.
     */
    Point2f offset;
    /*
     * The absolute offset between the correlated images. 
     */
    Point2f absoluteOffset;
    /*
     * The offset between the correlated images, in radians, 
     * based on a rotational projection model. 
     */
    Point2f angularOffset;
    /*
     * If valid is false, the reason why this correlation was
     * rejected. 
     */
    int rejectionReason;
    /*
     * The correlation coefficient of the correlation. 
     */
    double correlationCoefficient;

    double gainA;
    double gainB;

    /*
     * Creates a new instance of this class. 
     */
	CorrelationDiff() : 
        valid(false), 
        offset(0, 0), 
        angularOffset(0, 0), 
        rejectionReason(-1),
        correlationCoefficient(0) { }
};

/*
 * Class capable of correlating image pairs. 
 * Takes perspective into account. 
 */
class PairwiseCorrelator {

private:
    static const bool debug = false;
    /*
     * Definition of the underlying planar aligner to use. 
     */
    typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> Aligner;
public:
    PairwiseCorrelator() {
        AssertFalseInProduction(debug);
    }

    /*
     * Correlation was rejected, but to unknown reasons. Do not use. 
     */
    static const int RejectionUnknown = -1;
    /*
     * Correlation was not rejected. 
     */
    static const int RejectionNone = 0;
    /*
     * Correlation was rejected because images did not overlap. 
     */
    static const int RejectionNoOverlap = 1;
    /*
     * Correlation was rejected because deviation was to high. 
     */
    static const int RejectionDeviationTest = 2;
    /*
     * Correlation was rejected because the result was
     * too close to the image border. This is an indicator for
     * bad matching. 
     */
    static const int RejectionOutOfWindow = 3;
  
    /*
     * Matches two immages using a correlation (pixel) based aligner. The estimated perspective 
     * of the two images is taken into account. First, the overlapping region is recovered, then
     * the region is aligned.  
     *
     * @param a The first image. 
     * @param b The second image. 
     *
     * @param minWidth Minimal width of the overlapping region. 
     * @param minHeight Minimal height of the overlapping region. 
     *
     * @param forceWholeImage If true, forces usage of the whole image, 
     *                        even if the overlapping area is smaller.
     * @param w Correlation window size, relartive to the size of the image. Default 
     *          is 0.5, which lets the correlator try all positions up to 0.5 * width or 0.5 * heigth 
     *          away from the center. 
     *
     * @param wTolerance Additional tolerance to apply when checing for out-of-window correlations. 
     */ 
    CorrelationDiff Match(const InputImageP a, const InputImageP b, int minWidth = 0, int minHeight = 0, bool forceWholeImage = false, float w = 0.5, float wTolerance = 1) {

        AssertFalseInProduction(debug);

        // Flags used for global configuration of outlier tests. 
        const bool enableDeviationTest = false;
        const bool enableOutOfWindowTest = true;

        STimer cTimer(false);
        CorrelationDiff result;
        
        Mat wa, wb;

        cv::Point appliedBorder;

        cv::Point locationDiff = GetOverlappingRegion(a, b, a->image, b->image, wa, wb, a->image.cols * 0.2, appliedBorder);

        cTimer.Tick("Getting overlapping region");

        // Forces to use the whole image instead of predicted overlays.
        // Good for ring closure. We still have to guess the offset tough. 
        if(forceWholeImage) {
            appliedBorder = -locationDiff;
            wb = b->image.data;
            wa = a->image.data;
        }

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

        Mat corr; //Debug image used to print the correlation result.  

        PlanarCorrelationResult res = Aligner::Align(wa, wb, corr, w, w, 0);

        cTimer.Tick("Finding Correlation");

        int maxX = max(wa.cols, wb.cols) * w * wTolerance;
        int maxY = max(wa.rows, wb.rows) * w * wTolerance;

        if(enableOutOfWindowTest && (res.offset.x < -maxX || res.offset.x > maxX 
                || res.offset.y < -maxY || res.offset.y > maxY)) {

            if(debug) {
                cout << "Rejected because the correlation found an extremum at the border of the image." << endl;
            }

            result.valid = false;
            result.rejectionReason = RejectionOutOfWindow;
            return result;
        }

        if(enableDeviationTest && (res.topDeviation < 1.5)) {
            result.valid = false;
            result.rejectionReason = RejectionDeviationTest;
            
            if(debug) {
                cout << "Rejected because top deviation == " << res.topDeviation << " < 1.5." << endl;
            }

            return result;
        }

        cv::Point correctedRes = res.offset + appliedBorder;

        // Get hFov and vFov in radians. 
        // Calculate pixel per radian (linar vs. asin/atan)
        // We're working on the projectional tangent plane of B.
        
        double hFov = GetHorizontalFov(a->intrinsics);
        double vFov = GetVerticalFov(a->intrinsics);

        //cout << "hfov: " << hFov << ", vfov: " << vFov << endl;
        Point2d relativeOffset = 
            Point2d((double)correctedRes.x / a->image.cols, 
                    (double)correctedRes.y / a->image.rows); 

        //cout << "RelativeOffset: " << relativeOffset << endl;
        result.overlap = wa.cols * wa.rows; 

        // Careful! Rotational axis are swapped (movement along x axis corresponds to a rotation AROUND y axis) 
        result.angularOffset.y = asin(relativeOffset.x * sin(hFov));
        result.angularOffset.x = asin(relativeOffset.y * sin(vFov));
        result.offset = correctedRes;
        result.absoluteOffset = correctedRes + locationDiff;
        result.valid = true;
        result.correlationCoefficient = sqrt(res.variance) / res.n;
        result.gainA = res.gainA;
        result.gainB = res.gainB;
        
        cTimer.Tick("Estimating angular correlation");

        return result;
    }
};
}

#endif
