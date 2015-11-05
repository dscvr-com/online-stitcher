
#ifndef OPTONAUT_PLANAR_CORRELATOR_HEADER
#define OPTONAUT_PLANAR_CORRELATOR_HEADER

using namespace cv;
using namespace std; 

namespace optonaut {

template <typename Correlator>
class BruteForcePlanarAligner {
    public:
    static inline Point Align(const Mat &a, const Mat &b, Mat &corr, double wx = 1, double wy = 1) {
        int mx = 0;
        int my = 0;
        int w = max(a.cols, b.cols);
        int h = max(a.rows, b.rows);
        float min = std::numeric_limits<float>::max();

        corr = Mat(wy * h * 2, wx * w * 2, CV_32F);

        for(int dx = -w * wx; dx < w * wx; dx++) {
            for(int dy = -h * wy; dy < h * wy; dy++) {
                float res = Correlator::Calculate(a, b, dx, dy);
                corr.at<float>(dy + wy * h, dx + wx * w) = res; 

                if(res < min) {
                    min = res;
                    mx = dx;
                    my = dy;
                }
        
                //cout << dx << ", " << dy << endl;
            }
        }


        return Point(mx, my);
    }
};

template <typename ErrorMetric>
class BaseCorrelator {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int dx, int dy) {
        int sx = max(0, -dx);
        int ex = min(a.cols, b.cols - dy);
        
        int sy = max(0, -dy);
        int ey = min(a.rows, b.rows - dy);

        float corr = 0;

        for(int x = sx; x < ex; x++) {
            for(int y = sy; y < ey; y++) {
                 corr += ErrorMetric::Calculate(a, b, x, y, x + dx, y + dy);
            }
        }

        return corr;
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
};

template <typename T>
class AbsoluteDifference {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = ((float)a.at<T>(ya, xa) - (float)b.at<T>(yb, xb));
        return abs(diff);
    }
};

template <typename T>
class LeastSquares {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = AbsoluteDifference<T>::Calculate(a, b, xa, ya, xb, yb);
        return diff * diff;
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
};

template <typename T, int alpha>
class CrossCorrelation {
    public:
    static inline float Calculate(const Mat &a, const Mat &b, int xa, int ya, int xb, int yb) {
        float diff = ((float)a.at<T>(ya, xa) - alpha) * ((float)b.at<T>(yb, xb) - alpha);
        return diff;
    }
};
}
#endif
