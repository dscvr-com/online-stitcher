#include <memory>

#include "../common/support.hpp"
#include "../common/static_timer.hpp"
#include "../common/drawing.hpp"
#include "../math/stat.hpp"
#include "../common/assert.hpp"
#include "../stitcher/simplePlaneStitcher.hpp"

#ifndef OPTONAUT_PLANAR_CORRELATOR_HEADER
#define OPTONAUT_PLANAR_CORRELATOR_HEADER

using namespace cv;
using namespace std; 

namespace optonaut {

/*
 * Correlator debug flag - switch on for pairwise debug output. 
 */ 
static const bool debugCorrelator = false;
static const bool outputMatch = false;
static const bool assertsInLoopsOn = false;

/*
 * Represents the result of a planar correlation operation. 
 */
struct PlanarCorrelationResult {
    /*
     * The offset between the two images.
     */
    cv::Point offset;
    /*
     * The count of pixels in the overlapping area. 
     */
    size_t n; 
    /*
     * The error for the given correlation. 
     */
    double cost;
    /*
     * The variance for all the correlated pixels.
     */
    double variance;
    /*
     * The deviation for all the correlated pixels. 
     */
    double topDeviation;
};

/*
 * Finds a position with maximum correlation 
 * by trying all the possible positions. 
 *
 * @tparam Correlator The correlator function to use, 
 */ 
template <typename Correlator>
class BruteForcePlanarAligner {
    public:

    /*
     * Alignes to given images. 
     *
     * @param a The first image.
     * @param b The second image. 
     * @param corr The correlation result, just for debugging purposes. 
     * @param wx The correlation window in x direction.
     * @param wy The correlation window in y direction. 
     */
    static inline PlanarCorrelationResult Align(const Mat &a, const Mat &b, Mat &corr, double wx = 0.5, double wy = 0.5) {
        return Align(a, b, corr, max(a.cols, b.cols) * wx, max(a.rows, b.rows) * wy, 0, 0);
    }
    
    /*
     * Alignes to given images. 
     *
     * @param a The first image.
     * @param b The second image. 
     * @param corr The correlation result, just for debugging purposes. 
     * @param wx The correlation window in x direction.
     * @param wy The correlation window in y direction. 
     * @param ox The predefined offset in x direction.
     * @param oy The predefined offset in y direction. 
     */
    static inline PlanarCorrelationResult Align(const Mat &a, const Mat &b, Mat &corr, int wx, int wy, int ox, int oy) {
        STimer cTimer(false);

        int mx = 0;
        int my = 0;
        float min = std::numeric_limits<float>::max();
        OnlineVariance<double> var;

        if(debugCorrelator) {
            corr = Mat(wy * 2 + 1, wx * 2 + 1, CV_32F);
	        corr.setTo(Scalar::all(0));
        }

        if(assertsInLoopsOn) {
            AssertGTM(wx, 0, "Correlation window exists.");
            AssertGTM(wy, 0, "Correlation window exists.");
        }

        float costSum = 0;

        // Iterate over all possible offsets in our window. 
        for(int dx = -wx; dx <= wx; dx++) {
            for(int dy = -wy; dy <= wy; dy++) {
                
                // Run correlation for each position. 
                float res = Correlator::Calculate(a, b, dx + ox, dy + oy);

                // Collect results (add them to variance and cost caluclations) 
                var.Push(res);
                costSum += res;
                if(debugCorrelator) {
                    AssertGTM(corr.rows, dy + wy, "Correlation matrix too small.");
                    AssertGTM(corr.cols, dx + wx, "Correlation matrix too small.");
                    corr.at<float>(dy + wy, dx + wx) = res; 
                }

                // If we found a new minimum, remember it. 
                if(res < min) {
                    min = res;
                    mx = dx;
                    my = dy;
                }
            }
        }

        cTimer.Tick("BF correlator step - wx: " + ToString(wx) + ", wy: " + ToString(wy));

        double variance = var.Result();
        size_t count = static_cast<size_t>(wx * 2 + wy * 2);

        return { cv::Point(mx + ox, my + oy), count, costSum, variance, sqrt(variance / count)};
    }
};

/*
 * Finds a position with maximum correlation 
 * by using a pyramid correlation scheme. Downsampled versions
 * of the image are compared first, to constrain the window. 
 *
 * @tparam Correlator The correlator function to use, 
 */ 
template <typename Correlator>
class PyramidPlanarAligner {
    private:

    /*
     * Internal alignment function. Performs a recursive alignment step. 
     */
    static inline cv::Point AlignInternal(const Mat &a, const Mat &b, Mat &corr, int &corrXOff, int corrYOff, double wx, double wy, int dskip, int depth, VariancePool<double> &pool) {
        const int minSize = 4;

        cv::Point res;
        AssertFalseInProduction(debugCorrelator);

        if(a.cols > minSize / wx && b.cols > minSize / wx
                && a.rows > minSize / wy && b.rows > minSize / wy) {
            // If the image is large enough, perform a further pyramid alignment step. 
            Mat ta, tb;

            STimer cPyrDownTimer(false);
            pyrDown(a, ta);
            pyrDown(b, tb);
            cPyrDownTimer.Tick("Aligner PyrDown");

            cv::Point guess = PyramidPlanarAligner<Correlator>::AlignInternal(ta, tb, corr, corrXOff, corrYOff, wx, wy, dskip - 1, depth + 1, pool);

            if(debugCorrelator) {
                pyrUp(corr, corr);
            }

            if(dskip > 0) {
                // If we skip, just upsample the guess. 
                res = guess * 2;
            } else {

                STimer cTimer(false);
                Mat corrBf;

                // Perform a brute force correlation, but just for a very small area. 
                PlanarCorrelationResult detailedRes = 
                    BruteForcePlanarAligner<Correlator>::Align(
                            a, b, corrBf, 2, 2, guess.x * 2, guess.y * 2);

                if(debugCorrelator) {

                    cv::Rect roi(guess.x * 2 - corrBf.cols / 2 + corr.cols / 2 + corrXOff,
                             guess.y * 2 - corrBf.rows / 2 + corr.rows / 2 + corrYOff,
                             corrBf.cols, corrBf.rows);

                    corrXOff -= min(roi.x, 0),
                    corrYOff -= min(roi.y, 0),
                    copyMakeBorder(corr,
                                   corr,
                                   -min(roi.y, 0),
                                   -min(corr.rows - (roi.y + roi.height), 0),
                                   -min(roi.x, 0),
                                   -min(corr.cols - (roi.x + roi.width), 0),
                                   BORDER_CONSTANT, Scalar::all(0));
                    
                    roi.y = max(roi.y, 0);
                    roi.x = max(roi.x, 0);


                    corrBf.copyTo(corr(roi));
                }

                res = detailedRes.offset;
                auto weight = pow(2, depth);
                pool.Push(detailedRes.variance, detailedRes.n * weight, 
                        detailedRes.cost * weight);
                cTimer.Tick("BF Alignment step");
            }
        } else {
            // If we are at the bottom of our pyramid and the image is already very small, 
            // perform a brute-force correlation. 
            STimer cTimer(false);
            PlanarCorrelationResult detailedRes = 
                BruteForcePlanarAligner<Correlator>::Align(a, b, corr, wx, wy);
                
            res = detailedRes.offset;
            auto weight = pow(2, depth);
            pool.Push(detailedRes.variance, detailedRes.n * weight, 
                    detailedRes.cost * weight);
            cTimer.Tick("Lowest BF Alignment step");
        }
        
        return res;
    }
    public:
    
    /*
     * Alignes to given images. 
     *
     * @param a The first image.
     * @param b The second image. 
     * @param corr The correlation result, just for debugging purposes. 
     * @param wx The correlation window in x direction.
     * @param wy The correlation window in y direction.
     * @param dskip Skips dskip correlation steps from the top.  
     */
    static inline PlanarCorrelationResult Align(const Mat &a, const Mat &b, Mat &corr, double wx = 0.5, double wy = 0.5, int dskip = 0) {
        VariancePool<double> pool;
        int corrXOff = 0;
        int corrYOff = 0;
        AssertFalseInProduction(debugCorrelator);
        AssertFalseInProduction(outputMatch);

        // Invoke the internal alignment operation.
        cv::Point res = PyramidPlanarAligner<Correlator>::AlignInternal(a, b, corr, corrXOff, corrYOff, wx, wy, dskip, 0, pool);

        // Debug - draw the resulting image pair and correlation. 
        if(outputMatch) {
            static int dbgctr = 0;
            Mat eye = Mat::eye(3, 3, CV_64F);
            Mat hom = Mat::eye(3, 3, CV_64F);
            hom.at<double>(0, 2) = res.x;
            hom.at<double>(1, 2) = res.y;

            SimplePlaneStitcher stitcher;

            auto target = stitcher.Stitch(
                    {std::make_shared<Image>(Image(a)), 
                    std::make_shared<Image>(Image(b))}, 
                    {res, cv::Point(0, 0)});

            //DrawMatchingResults(hom, eye, a, b, target);
            
            std::string filename =  
                        "match_" + 
                        ToString(dbgctr) + "_" +  
                        ToString(sqrt(pool.GetMeasurements().back().s) / 
                            pool.GetMeasurements().back().n);

            imwrite("dbg/" + filename + ".jpg", target->image.data);
            /*
            float max = 255;

            for(int i = 0; i < corr.cols; i++) {
                for(int j = 0; j < corr.rows; j++) {
                    if(corr.at<float>(j, i) > max) {
                        max = corr.at<float>(j, i);
                    }
                }
            }
            
            imwrite("dbg/" + filename + "_corr.jpg", corr / max * 255);
            */
            dbgctr++;
        }
        
        double topDeviation = sqrt(pool.GetMeasurements().back().s) / 
                            pool.GetMeasurements().back().n;

        return { res, pool.Count(), pool.Sum(), pool.Result(), topDeviation };
    }
};

/*
 * Base correlator. Just sums up the pixel-wise errors
 * for each overlapping region. 
 */
template <typename ErrorMetric>
class BaseCorrelator {
    public:
    /*
     * Calculates the correlation value.
     * 
     * @param a The first image.
     * @param b The second image.
     * @param dx Offset in x direction.
     * @param dy Offset in y direction. 
     */
    static inline float Calculate(const Mat &a, const Mat &b, int dx, int dy) {

        // Get overlapping area.
        int sx = max(0, -dx);
        int ex = min(a.cols, b.cols - dx);
        
        int sy = max(0, -dy);
        int ey = min(a.rows, b.rows - dy);

        float corr = 0;

        // Lop over overlapping area and calculate correlation value. 
        for(int x = sx; x < ex; x++) {
            for(int y = sy; y < ey; y++) {
                 corr += ErrorMetric::Calculate(a, b, x, y, x + dx, y + dy);
            }
        }

        return corr * ErrorMetric::Sign();
    }
};

/*
 * Normed correlator. Norms the pixel-wise error sum according
 * to the overlapping area. 
 */
template <typename ErrorMetric>
class NormedCorrelator {
    public:
    /*
     * Calculates the correlation value.
     * 
     * @param a The first image.
     * @param b The second image.
     * @param dx Offset in x direction.
     * @param dy Offset in y direction. 
     */
    static inline float Calculate(const Mat &a, const Mat &b, int dx, int dy) {
        float sx = max(0, -dx);
        float ex = min(a.cols, b.cols - dx);
        
        float sy = max(0, -dy);
        float ey = min(a.rows, b.rows - dy);

        // Get base correlation value. 
        float corr = BaseCorrelator<ErrorMetric>::Calculate(a, b, dx, dy);
        // Divide by area. 
        return corr / ((ex - sx) * (ey - sy));
    }
    static inline float Sign() {
        return 1;
    }
};

/*
 * Error metric - absolute difference of pixel values. 
 */
template <typename T>
class AbsoluteDifference {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = ((float)a.at<T>(ya, xa) - (float)b.at<T>(yb, xb));
        return abs(diff);
    }
    static inline float Sign() {
        return 1;
    }
};

/*
 * Error metric - absolute difference of pixel values. 
 * Implementation for color images. 
 */
template <>
class AbsoluteDifference<Vec3b> {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        uchar *va = a.data + (ya * a.cols + xa) * 3;
        uchar *vb = b.data + (yb * b.cols + xb) * 3;
       // auto va = a.at<cv::Vec3b>(ya, xa);
       // auto vb = b.at<cv::Vec3b>(yb, xb);

        float db = (float)va[0] - (float)vb[0];
        float dr = (float)va[1] - (float)vb[1];
        float dg = (float)va[2] - (float)vb[2];
                
        return (std::abs(db) + std::abs(dr) + std::abs(dg)) / 3;
    }
    static inline float Sign() {
        return 1;
    }
};

/*
 * Error metric - squared difference of pixel values. 
 */
template <typename T>
class LeastSquares {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = AbsoluteDifference<T>::Calculate(a, b, xa, ya, xb, yb);
        return (diff * diff);
    }
    static inline float Sign() {
        return 1;
    }
};


/*
 * Error metric - squared difference of pixel values.
 * Overload for color images.  
 */
template <>
class LeastSquares<Vec3b> {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        auto va = a.at<cv::Vec3b>(ya, xa);
        auto vb = b.at<cv::Vec3b>(yb, xb);

        float db = (float)va[0] - (float)vb[0];
        float dr = (float)va[1] - (float)vb[1];
        float dg = (float)va[2] - (float)vb[2];
                
        return (db * db + dr * dr + dg * dg) / (3 * 3);
    }
    static inline float Sign() {
        return 1;
    }
};

/*
 * Error metric - GemanMcClure metric. 
 */
template <typename T, int alpha>
class GemanMcClure {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float x = AbsoluteDifference<T>::Calculate(a, b, xa, ya, xb, yb);
        x = x * x;
        return (x) / (1.0 + x / (alpha * alpha));
    }
    static inline float Sign() {
        return 1;
    }
};

/*
* Error metric - cross correlation. Only works for floating-point based images
* that are centered around some average (alpha).  
*/
template <typename T, int alpha>
class CrossCorrelation {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = ((float)a.at<T>(ya, xa) - alpha) * ((float)b.at<T>(yb, xb) - alpha);
        return diff;
    }
    static inline float Sign() {
        return -1;
    }
};

/*
* Error metric - cross correlation. Only works for floating-point based images
* that are centered around some average (alpha). Overload for color images. 
*/
template <int alpha>
class CrossCorrelation<Vec3b, alpha> {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        static const Vec3b alphaV = Vec3b(alpha, alpha, alpha);
        float diff = (a.at<Vec3b>(ya, xa) - alphaV).ddot(b.at<Vec3b>(yb, xb) - alphaV);
        return diff / (256 * 3);
    }
    static inline float Sign() {
        return -1;
    }
};
}
#endif
