#include <algorithm>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "../math/projection.hpp"
#include "../stitcher/simpleSphereStitcher.hpp"

#ifndef OPTONAUT_DRAWING_DEBUG_HEADER
#define OPTONAUT_DRAWING_DEBUG_HEADER

using namespace cv;
using namespace cv::detail;
using namespace std;

namespace optonaut {

    /*
     * Draws a polygon, given as a list of points. 
     *
     * @param target The image to draw to.
     * @param corners The corners of the polygon to draw. 
     * @param color The color to use for drawing (optional). 
     */
    static inline void DrawPoly(const Mat &target, const vector<Point2f> &corners, const Scalar color = Scalar(255, 0, 0)) {
        
        Point2f last = corners.back();

        for(auto point : corners) {
            line(target, last, point, color, 4);
            last = point;
        }
    }

    /*
     * Draws a rectangle.
     *
     * @param target The image to draw to.
     * @param roi The rectangle to draw.
     * @param color The color to use for drawing (optional).
     */
    static inline void DrawBox(const Mat &target, const cv::Rect &roi, const Scalar color = Scalar(255, 0, 0)) {
        std::vector<Point2f> corners;
        corners.emplace_back(roi.x, roi.y);
        corners.emplace_back(roi.x, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y);

        DrawPoly(target, corners, color);
    }

    
    /*
     * Draws a bar.
     *
     * @param image The image to draw to.
     * @param color The color to use for drawing (optional).
     */
    static inline void DrawBar(cv::Mat &image, double val) {
        Scalar color;
        if(val < 0) {
            color = Scalar(0, 255, 255);
        } else {
            color = Scalar(255, 0, 255);
        }
        cv::rectangle(image, cv::Point(0, image.rows * (0.5 - val)), cv::Point(image.cols, image.rows * 0.5), color, CV_FILLED);
    }
    
    
    /*
     * Draws the border of another image related by a homography. 
     *
     * @param homography The homography that relates the image. 
     * @param left The image that is releated by the homography. 
     * @param color The color to use for drawing (optional).
     * @param target The image to draw to.
     */
    static inline void DrawMatchingHomographyBorder(const Mat &homography, const Mat &left, const Scalar &color, Mat &target) {
        std::vector<Point2f> scene_corners = GetSceneCorners(left, homography);

        Point2f offset(left.cols, 0);

        for(size_t i = 0; i < scene_corners.size(); i++) {
            scene_corners[i] += offset;
        }

        DrawPoly(target, scene_corners, color);
    }

    /*
     * Draws matching results from two different homographies. 
     *
     * @param homography The homography that relates the image. 
     * @param homographyFromRot The second homography that relates the image. 
     * @param a The image that is releated by the homography. 
     * @param b The second image that is releated by the homography. 
     * @param target The image to draw to, usually the same as a.
     */
    static inline void DrawMatchingResults(const Mat &homography, const Mat &homographyFromRot, const Mat &a, const Mat &b, Mat &target) {
        
        a.copyTo(target(cv::Rect(0, 0, a.cols, a.rows)));
        b.copyTo(target(cv::Rect(a.cols, 0, b.cols, b.rows)));

        DrawMatchingHomographyBorder(homography, a, Scalar(0, 255, 0), target);
        DrawMatchingHomographyBorder(homographyFromRot, a, Scalar(255, 0, 0), target);
    }

    /*
     * Draws points given in a rotational model onto an equirectangular panorama. 
     *
     * @param target The image to draw to.
     * @param positions The positions, given in rotational coordinates. 
     * @param intrinsics The camera intrinsics associated with the images on the panorama. 
     * @param imageSize The size of images in the panorama. 
     * @param warperScale Scale of the cv::Warper that applies coordinate transform. 
     * @param offset Offset to apply to target coordinates when drawing. 
     * @param color The color to use. 
     * @param size The size of the circle to draw. 
     */
    static inline void DrawPointsOnPanorama(Mat &target, const vector<Mat> &positions,
            const cv::Mat &intrinsics, const cv::Size &imageSize, const float warperScale,
            const cv::Point &offset,
            const Scalar color = Scalar(0x00, 0xFF, 0x00), const int size = 8) {

        SimpleSphereStitcher debugger(warperScale);

        for(auto ext : positions) {

            cv::Point center = debugger.WarpPoint(intrinsics,
                       ext, 
                       imageSize, cv::Point(0, 0));

            cv::circle(target, center - offset, size, color, -1);

            //if(text != "") {
            //    Point offset(-20, -20);
            //    
            //    cv::putText(target, text, 
            //            center + offset, FONT_HERSHEY_PLAIN, 3, 
            //            color, 3);
            //}
        }
    }
   
    /*
     * Draws the center of each given image to the target panorama. 
     *
     * @param target The panorama to draw to.
     * @param images The images to draw the centers for.  
     * @param warperScale Scale of the cv::Warper that applies coordinate transform. 
     * @param color The color to use. 
     */ 
    static inline void DrawImagePointsOnPanorama(StitchingResultP &target, const vector<InputImageP> &images, 
            float warperScale, const Scalar color = Scalar(0x00, 0xFF, 0x00)) {
       DrawPointsOnPanorama(target->image.data, 
                ExtractExtrinsics(images), images[0]->intrinsics, 
                images[0]->image.size(), warperScale, target->corner, color);
    }

}

#endif

