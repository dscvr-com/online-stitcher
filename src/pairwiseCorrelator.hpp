#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "support.hpp"
#include "projection.hpp"
#include "drawing.hpp"
#include "stat.hpp"

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

	CorrelationDiff() : valid(false), offset(0, 0) {}

};

class PairwiseCorrelator {

private:
    static const bool debug = false;
    static const bool drawDebugBorders = false;

    void ReduceForCorrelation(const Mat &in, Mat &out) {
        vector<size_t> reduced(in.cols);
        size_t max = 1;

        for(int i = 0; i < in.cols; i++) {
            reduced[i] = 0;
            for(int j = 0; j < in.rows; j++) {
                reduced[i] += in.at<uchar>(j, i);
            }
            if(max < reduced[i]) {
                max = reduced[i];
            }
        }
        
        out = Mat(10, in.cols, CV_8U);
        for(int i = 0; i < in.cols; i++) {
            uchar val = (reduced[i]) / in.rows;
            for(int j = 0; j < 10; j++) {
                out.at<uchar>(j, i) = val;
            }
        }
    }

    double ShittyCorrelation(const Mat &a, const Mat &b, Mat &corr, int &cx) {
        double best = -1;
       
        assert(a.rows == b.rows);
        
        int w = a.cols;
        const double window = 0.5;
        deque<double> cv(w * 2 * window + 1);
        unsigned char *ad = (unsigned char*)(a.data);
        unsigned char *bd = (unsigned char*)(b.data);

        for(int i = -w * window; i < w * window; i++) {
          
            double cur = 0;
            for(int j = 0; j < w; j++) {
                int xa = j;
                int xb = j + i;

                if(xb > 0 && xb < w) {
                    for(int q = 0; q < a.rows; q += 4) { 
                        //Correlation
                        cur += (double)ad[q * a.cols + xa] * (double)bd[q * b.cols + xb];
                        //cur += (a.at<uchar>(q, xa)) * (int)(b.at<uchar>(q, xb));
                        //Shitty Norm 
                        //cur += abs((int)a.at<uchar>(0, xa) - (int)b.at<uchar>(0, xb));
                    }
                }
           }
           cur = cur / (w - abs(i)) / (a.rows / 4);
           cv[i + w * window] = cur;
           if(cur > best) {
               best = cur;
               cx = i;
           }
       }
       if(debug) {
            corr = Mat(10, w * 2 * window + 1, CV_8U);
            for(int j = 0; j < w; j++) {
                for(int i = 0; i < 10; i++) {
                    corr.at<uchar>(i, j) = cv[j] * 255.0 / best;
                }
            }
       }

        double var = Deviation(cv);
        
        return var;
    }
public:

    CorrelationDiff Match(const ImageP a, const ImageP b) {

        const bool useGradient = false;
        const bool useReduce = true;

        CorrelationDiff result;
        Mat ga, gb;
        cvtColor(a->img, ga, CV_BGR2GRAY);
        cvtColor(b->img, gb, CV_BGR2GRAY); 

        if(useGradient || useReduce) {
            Mat fa, fb;
            fa = ga;
            fb = gb;
            //double rad = 15;
            //medianBlur(ga, fa, rad);
            //medianBlur(gb, fb, rad);
            GetGradient(fa, ga, 1, 0);
            GetGradient(fb, gb, 1, 0);
        }

        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);

        HomographyFromImages(a, b, hom, rot);
        
        Mat wa(ga.rows, ga.cols, CV_64F);
        warpPerspective(ga, wa, hom, wa.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
       
        ga = wa;
        
        //Cut images, set homography to id.
        vector<Point2f> corners = GetSceneCorners(ga, hom);
        cv::Rect roi = GetInnerBoxForScene(corners);
        
        if(drawDebugBorders) {
            DrawBox(ga, roi, Scalar(0x70));
            DrawBox(gb, roi, Scalar(0x70));
        }

        roi = roi & cv::Rect(0, 0, ga.cols, ga.rows);
        
        if(drawDebugBorders) {
            DrawPoly(ga, corners, Scalar(0xc0));
            DrawPoly(gb, corners, Scalar(0xc0));
            DrawBox(ga, roi, Scalar(255));
            DrawBox(gb, roi, Scalar(255));
        }

        hom = Mat::eye(3, 3, CV_32F);
        ga = ga(roi);
        gb = gb(roi);

        //If those asserts fire, we've fed the aligner two non-overlapping 
        //images probably. SHAME!
        if(roi.width < 1 || roi.height < 1) {
            return result;
        }
        
        Mat corr(0, 0, CV_8U); 
        Mat ca(0, 0, CV_8U);
        Mat cb(0, 0, CV_8U);
        double var = 0;
        
        if(useReduce) {
            //ReduceForCorrelation(ga, ca);
            //ReduceForCorrelation(gb, cb);
            int dx; 
            var = ShittyCorrelation(ga, gb, corr, dx);
            result.offset = Point2f(dx, 0);
            result.valid = var > 50;
            //ga = ca;
            //gb = cb;
        } else {
            const int warp = MOTION_AFFINE;
            Mat affine = Mat::eye(2, 3, CV_32F);

            const int iterations = 100;
            const double eps = 1e-5;

            TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
            try {
                findTransformECC(ga, gb, affine, warp, termination);
                result.offset = Point2f(affine.at<float>(0, 2), affine.at<float>(1, 2));
                result.valid = true;
            } catch (Exception ex) {
                result.valid = false;
            }
        }


        //debug - draw resulting image. 
        if(debug) {
            Mat eye = Mat::eye(3, 3, CV_64F);
            Mat hom = Mat::eye(3, 3, CV_64F);
            hom.at<double>(0, 2) = result.offset.x;
            hom.at<double>(1, 2) = result.offset.y;
            Mat target(ga.rows, ga.cols + gb.cols, CV_8UC3);

            vector<DMatch> dummy;

            DrawMatchingResults(hom, eye, dummy, ga, ImageFeatures(), gb, ImageFeatures(), target);
            std::string filename =  
                    "dbg/ecc_result" + ToString(a->id) + 
                    " " + ToString(b->id) + 
                    " x-corr " + ToString(result.offset.x) + 
                    " var " + ToString(var) + 
                    " .jpg";
            if(useReduce) {
                cvtColor(corr, corr, CV_GRAY2BGR);
                corr.copyTo(target(Rect((target.cols - corr.cols) / 2, corr.rows, corr.cols, corr.rows))); 
                //cvtColor(ca, ca, CV_GRAY2BGR);
                //ca.copyTo(target(Rect(0, 0, ca.cols, ca.rows))); 
                //cvtColor(cb, cb, CV_GRAY2BGR);
                //cb.copyTo(target(Rect(ca.cols, 0, ca.cols, ca.rows))); 
            }
            imwrite(filename, target);
        }

        return result;
    }
};
}

#endif
