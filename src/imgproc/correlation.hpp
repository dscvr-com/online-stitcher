#include <opencv2/opencv.hpp>

#include "../math/stat.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_CORRELATION_HEADER
#define OPTONAUT_CORRELATION_HEADER

namespace optonaut {

    static inline double CorrelateX(const Mat &a, const Mat &b, double window, int &dx, Mat &corr, double bias = 0, int biasX = 0, double biasWidth = 0, bool debug = true) {
        
        double best = -1;
        double worst = std::numeric_limits<double>::max();
        double cur = 0;
       
        int w = min(a.cols, b.cols);
        const int skip = 1;
        unsigned char *ad = (unsigned char*)(a.data);
        unsigned char *bd = (unsigned char*)(b.data);
        
        deque<double> values(w * 2 * window + 1);

        for(int i = -w * window; i < w * window; i++) {
            cur = 0;

            for(int j = 0; j < w; j++) {
                int xa = j;
                int xb = j + i;

                if(xb > 0 && xb < w) {
                    for(int q = 0; q < min(a.rows, b.rows); q += skip) { 
                        cur += (double)ad[q * a.cols + xa] * (double)bd[q * b.cols + xb];
                    }
                }
           }
           cur = cur / (w - abs(i)) / (a.rows / skip);
           
           if(bias != 0) {
               cur += gauss(i, bias, biasX, biasWidth);
           }

           values[i + w * window] = cur;

           if(cur > best) {
               best = cur;
               dx = i;
           }
           if(cur < worst) {
               worst = cur;
           }
       }

       if(debug) {
            corr = Mat(10, w * 2 * window + 1, CV_8U);
            for(int j = 0; j < corr.cols; j++) {
                for(int i = 0; i < 10; i++) {
                    corr.at<uchar>(i, j) = (values[j] - worst) * 255.0 / (best - worst);
                }
            }
       }

        double var = Deviation(values);
        
        return var;
    }
    
    static inline double CorrelateX(const Mat &a, const Mat &b, double window, int &dx) {
        Mat dum;
        return CorrelateX(a, b, window, dx, dum, false);
    }

    static inline double CorrelateY(const Mat &a, const Mat &b, double window, int &dy, Mat &corr, bool debug = true) {
        
        double best = -1;
        double cur = 0;
       
        int h = min(a.rows, b.rows);
        const int skip = 4;
        unsigned char *ad = (unsigned char*)(a.data);
        unsigned char *bd = (unsigned char*)(b.data);
        
        deque<double> values(h * 2 * window + 1);

        for(int i = -h * window; i < h * window; i++) {
            cur = 0;

            for(int j = 0; j < h; j++) {
                int xa = j;
                int xb = j + i;

                if(xb > 0 && xb < h) {
                    for(int q = 0; q < min(a.cols, b.cols); q += skip) { 
                        cur += (double)ad[xa * a.cols + q] * (double)bd[xb * b.cols + q];
                    }
                }
           }
           cur = cur / (h - abs(i)) / (a.cols / skip);
           values[i + h * window] = cur;

           if(cur > best) {
               best = cur;
               dy = i;
           }
       }

       if(debug) {
            corr = Mat(h * 2 * window + 1, 10, CV_8U);
            for(int j = 0; j < h; j++) {
                for(int i = 0; i < 10; i++) {
                    corr.at<uchar>(j, i) = values[i] * 255.0 / best;
                }
            }
       }

        double var = Deviation(values);
        
        return var;
    }

    static inline double CorrelateY(const Mat &a, const Mat &b, double window, int &cy) {
        Mat dum;
        return CorrelateY(a, b, window, cy, dum, false);
    }
}
#endif
