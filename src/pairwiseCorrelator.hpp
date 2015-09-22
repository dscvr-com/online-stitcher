#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "support.hpp"
#include "io.hpp"
#include "projection.hpp"
#include "drawing.hpp"

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
    static const bool debug = true;
public:

    CorrelationDiff Match(const ImageP a, const ImageP b) {

        const bool useGradient = false;
        const bool useReduce = false;
        const bool debug = false;

        CorrelationDiff result;
        Mat ga, gb;
        cvtColor(a->img, ga, CV_BGR2GRAY);
        cvtColor(b->img, gb, CV_BGR2GRAY); 

        if(useGradient) {
            GetGradient(ga, ga);
            GetGradient(gb, gb);
        }

        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);

        HomographyFromImages(a, b, hom, rot);
        
        Mat wa(ga.rows, ga.cols, CV_64F);
        warpPerspective(ga, wa, hom, wa.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
       
        ga = wa;
        
        //Cut images, set homography to id.
        vector<Point2f> corners = GetSceneCorners(ga, hom);
        Rect roi = GetInnerBoxForScene(corners);
        
        if(debug) {
            DrawBox(ga, roi, Scalar(0x70));
            DrawBox(gb, roi, Scalar(0x70));
        }

        roi = roi & Rect(0, 0, ga.cols, ga.rows);
        
        if(debug) {
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
            assert(false);
        }

        if(useReduce) {
            reduce(ga, ga, 0, CV_REDUCE_AVG);
            reduce(gb, gb, 0, CV_REDUCE_AVG);
        }

        const int warp = MOTION_TRANSLATION;
        Mat affine = Mat::eye(2, 3, CV_32F);

        const int iterations = 1000;
        const double eps = 1e-5;

        TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
        try {
            findTransformECC(ga, gb, affine, warp, termination);
            
            result.offset = Point2f(affine.at<float>(0, 2), affine.at<float>(1, 2));
            result.valid = true;
        } catch (Exception ex) {
            result.valid = false;
        }

        //debug - draw resulting image. 
        if(debug) {
            Mat eye = Mat::eye(3, 3, CV_64F);
            Mat hom = Mat::eye(3, 3, CV_64F);
            hom.at<double>(0, 2) = result.offset.x;
            hom.at<double>(1, 2) = result.offset.y;
            Mat target(ga.rows, ga.cols + gb.cols, CV_8UC3);

            vector<DMatch> dummy;

            DrawMatchingResults(hom, eye, eye, ga, ImageFeatures(), gb, ImageFeatures(), target);
            std::string filename =  
                    "dbg/ecc_result" + ToString(a->id) + 
                    "_" + ToString(b->id) + 
                    "_x-corr " + ToString(affine.at<float>(0, 2)) + " .jpg";
            
            imwrite(filename, target);
        }

        return result;
    }
};
}

#endif
