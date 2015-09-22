#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_DRAWING_DEBUG_HEADER
#define OPTONAUT_DRAWING_DEBUG_HEADER

namespace optonaut {

    static inline void DrawPoly(const Mat &target, const vector<Point2f> &corners, const Scalar color = Scalar(255, 0, 0)) {
        
        Point2f last = corners.back();

        for(auto point : corners) {
            line(target, last, point, color, 4);
            last = point;
        }
    }

    static inline void DrawBox(const Mat &target, const Rect &roi, const Scalar color = Scalar(255, 0, 0)) {
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
    
}

#endif

