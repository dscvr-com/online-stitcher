#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_DRAWING_DEBUG_HEADER
#define OPTONAUT_DRAWING_DEBUG_HEADER

using namespace cv;
using namespace cv::detail;
using namespace std;

namespace optonaut {

    static inline void DrawPoly(const Mat &target, const vector<Point2f> &corners, const Scalar color = Scalar(255, 0, 0)) {
        
        Point2f last = corners.back();

        for(auto point : corners) {
            line(target, last, point, color, 4);
            last = point;
        }
    }

    static inline void DrawBox(const Mat &target, const cv::Rect &roi, const Scalar color = Scalar(255, 0, 0)) {
        std::vector<Point2f> corners;
        corners.emplace_back(roi.x, roi.y);
        corners.emplace_back(roi.x, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y);

        DrawPoly(target, corners, color);
    }

    
    static inline void DrawBar(cv::Mat &image, double val) {
        Scalar color;
        if(val < 0) {
            color = Scalar(0, 255, 255);
        } else {
            color = Scalar(255, 0, 255);
        }
        cv::rectangle(image, cv::Point(0, image.rows * (0.5 - val)), cv::Point(image.cols, image.rows * 0.5), color, CV_FILLED);
    }
    
    static inline void DrawMatchingHomographyBorder(const Mat &homography, const Mat &left, const Scalar &color, Mat &target) {
        std::vector<Point2f> scene_corners = GetSceneCorners(left, homography);

        Point2f offset(left.cols, 0);

        for(size_t i = 0; i < scene_corners.size(); i++) {
            scene_corners[i] += offset;
        }

        DrawPoly(target, scene_corners, color);
    }

    static inline void DrawMatchingResults(const Mat &homography, const Mat &homographyFromRot, const vector<DMatch> &goodMatches, const Mat &a, const ImageFeatures &aFeatures, const Mat &b, const ImageFeatures &bFeatures, Mat &target) {

        //Colors: Green: Detected Homography.
        //        Red:   Estimated from Sensor.
        //        Blue:  Hmoography induced by dectected rotation. 

  		drawMatches(a, aFeatures.keypoints, b, bFeatures.keypoints,
               goodMatches, target, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        DrawMatchingHomographyBorder(homography, a, Scalar(0, 255, 0), target);
        DrawMatchingHomographyBorder(homographyFromRot, a, Scalar(255, 0, 0), target);
       
        //Estimation not possible without imageP 
        //Mat estimation;
        //Mat rot;
        //HomographyFromKnownParameters(a, b, estimation, rot);
        //DrawHomographyBorder(estimation, a, Scalar(0, 0, 255), target);
    } 

}

#endif

