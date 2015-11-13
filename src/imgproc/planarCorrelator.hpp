#include "../common/support.hpp"
#include "../common/drawing.hpp"

#ifndef OPTONAUT_PLANAR_CORRELATOR_HEADER
#define OPTONAUT_PLANAR_CORRELATOR_HEADER

using namespace cv;
using namespace std; 

namespace optonaut {

static const bool debug = false;

template <typename Correlator>
class BruteForcePlanarAligner {
    public:
    static inline Point Align(const Mat &a, const Mat &b, Mat &corr, double wx = 0.5, double wy = 0.5) {
        return Align(a, b, corr, max(a.cols, b.cols) * wx, max(a.rows, b.rows) * wy, 0, 0);
    }
    
    static inline Point Align(const Mat &a, const Mat &b, Mat &corr, int wx, int wy, int ox, int oy) {
        int mx = 0;
        int my = 0;
        float min = std::numeric_limits<float>::max();

        if(debug) {
            corr = Mat(wy * 2, wx * 2, CV_32F);
	        corr.setTo(Scalar::all(0));
        }

        for(int dx = -wx; dx < wx; dx++) {
            for(int dy = -wy; dy < wy; dy++) {
                float res = Correlator::Calculate(a, b, dx + ox, dy + oy);
                if(debug) {
                    corr.at<float>(dy + wy, dx + wx) = res; 
                }
                if(res < min) {
                    min = res;
                    mx = dx;
                    my = dy;
                }
            }
        }


        return Point(mx + ox, my + oy);
    }
};

template <typename Correlator>
class PyramidPlanarAligner {
    public:
    static inline Point Align(const Mat &a, const Mat &b, double wx = 0.5, double wy = 0.5, int dskip = 0) {
        const int minSize = 4;

        Point res;
        Mat corr(0, 0, CV_32F);

        if(a.cols > minSize / wx && b.cols > minSize / wx 
                && a.rows > minSize / wy && b.rows > minSize / wy) {
            Mat ta, tb;

            pyrDown(a, ta);
            pyrDown(b, tb);

            Point guess = PyramidPlanarAligner<Correlator>::Align(ta, tb, wx, wy, dskip - 1);
            if(dskip > 0) {
                res = guess * 2;
            } else {
                res = BruteForcePlanarAligner<Correlator>::Align(a, b, corr, 2, 2, guess.x * 2, guess.y * 2);
            }
        } else {
            res = BruteForcePlanarAligner<Correlator>::Align(a, b, corr, wx, wy);
        }
        
        //debug - draw resulting image. 
        if(debug && dskip > 0) {
            static int dbgctr = 0;
            Mat eye = Mat::eye(3, 3, CV_64F);
            Mat hom = Mat::eye(3, 3, CV_64F);
            hom.at<double>(0, 2) = res.x;
            hom.at<double>(1, 2) = res.y;

            Mat target(max(max(a.rows, a.rows), corr.rows), corr.cols + a.cols + b.cols, CV_8UC3);

            DrawMatchingResults(hom, eye, a, b, target);
            std::string filename =  
                    "dbg/pcc_result_" + ToString(dbgctr) + 
                    "_x_corr_" + ToString(res.x) + 
                    ".jpg";
            Rect targetRoi(b.cols + a.cols, 0, corr.cols, corr.rows);
            cvtColor(corr, corr, CV_GRAY2BGR);
            corr.copyTo(target(targetRoi));

            imwrite(filename, target);
            dbgctr++;
        }

        return res;
    }
};

template <typename ErrorMetric>
class BaseCorrelator {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int dx, int dy) {
        int sx = max(0, -dx);
        int ex = min(a.cols, b.cols - dx);
        
        int sy = max(0, -dy);
        int ey = min(a.rows, b.rows - dy);

        float corr = 0;

        for(int x = sx; x < ex; x++) {
            for(int y = sy; y < ey; y++) {
                 corr += ErrorMetric::Calculate(a, b, x, y, x + dx, y + dy);
            }
        }

        return corr * ErrorMetric::Sign();
    }
};

template <typename ErrorMetric>
class NormedCorrelator {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int dx, int dy) {
        float sx = max(0, -dx);
        float ex = min(a.cols, b.cols - dx);
        
        float sy = max(0, -dy);
        float ey = min(a.rows, b.rows - dy);

        float corr = BaseCorrelator<ErrorMetric>::Calculate(a, b, dx, dy);
        return corr / ((ex - sx) * (ey - sy));
    }
    static inline float Sign() {
        return 1;
    }
};

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
                
        return (abs(db) + abs(dr) + abs(dg)) / 3;
    }
    static inline float Sign() {
        return 1;
    }
};

template <typename T>
class LeastSquares {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = AbsoluteDifference<T>::Calculate(a, b, xa, ya, xb, yb);
        return diff * diff;
    }
    static inline float Sign() {
        return 1;
    }
};

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
