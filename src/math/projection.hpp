#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#include "../io/inputImage.hpp"
#include "../common/image.hpp"
#include "../common/assert.hpp"
#include "../common/support.hpp"
#include "support.hpp"

#ifndef OPTONAUT_PROJECTION_HEADER
#define OPTONAUT_PROJECTION_HEADER

using namespace std;
using namespace cv;

namespace optonaut {
    static inline vector<Point2f> GetSceneCorners(const Image &img, const Mat &homography) {
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); 
        obj_corners[1] = cvPoint(img.cols, 0);
        obj_corners[2] = cvPoint(img.cols, img.rows); 
        obj_corners[3] = cvPoint(0, img.rows);
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, homography);

        return scene_corners;
    }

    static inline cv::Rect GetBoundingBox(const vector<Point2f> &points) {
        assert(points.size() > 0);

        double t = points[0].y;
        double b = points[0].y;
        double l = points[0].x;
        double r = points[0].x;

        for(auto p : points) {
            t = min2(p.y, t);
            b = max2(p.y, b);
            l = min2(p.x, l);
            r = max2(p.x, r);
        }

        return cv::Rect(l, t, r - l, b - t);
    }
    
    static inline cv::Rect GetInnerBoxForScene(const vector<Point2f> &c) {
        assert(c.size() == 4);

        double l = max(c[0].x, c[3].x);
        double t = max(c[0].y, c[1].y);
        double r = min(c[1].x, c[2].x);
        double b = min(c[2].y, c[3].y);

        return cv::Rect(l, t, r - l, b - t);
    }
    
    static inline void HomographyFromRotation(const Mat &rot, const Mat &k, Mat &hom) {
        hom = k * rot * k.inv();
    }

    static inline void RotationFromImages(const InputImageP a, const InputImageP b, Mat &rot) {
        rot = b->adjustedExtrinsics.inv() * a->adjustedExtrinsics;
    }

    static inline void HomographyFromImages(const InputImageP a, const InputImageP b, Mat &hom, Mat &rot, cv::Size scale) {
        Mat R3(3, 3, CV_64F);
        Mat aK3(3, 3, CV_64F);

        RotationFromImages(a, b, rot);
        
        From4DoubleTo3Double(rot, R3);

        ScaleIntrinsicsToImage(a->intrinsics, scale, aK3);

        HomographyFromRotation(R3, aK3, hom);

        hom.at<double>(0, 2) += rot.at<double>(0, 3);
        hom.at<double>(1, 2) += rot.at<double>(1, 3);

        AssertWEQM(rot.at<double>(0, 3), 0.0, "Given transformation is translating along x axis. Please double check result.");
        AssertWEQM(rot.at<double>(1, 3), 0.0, "Given transformation is translating along y axis. Please double check result.");

        AssertEQM(rot.at<double>(2, 3), 0.0, "Given transformation is translating along z axis. This is not supported for generating homographies.");
    }
    
    static inline void HomographyFromImages(const InputImageP a, const InputImageP b, Mat &hom, Mat &rot) {
        HomographyFromImages(a, b, hom, rot, a->image.size());
    }

    static inline bool ImagesAreOverlapping(const InputImageP a, const InputImageP b, double minOverlap = 0.1) {
        Mat hom, rot;

        HomographyFromImages(a, b, hom, rot);

        std::vector<Point2f> corners = GetSceneCorners(a->image, hom); 

        int top = min(corners[0].x, corners[1].x); 
        int bot = max(corners[2].x, corners[3].x); 
        int left = min(corners[0].y, corners[3].y); 
        int right = max(corners[1].y, corners[2].y); 

        int x_overlap = max(0, min(right, b->image.cols) - max(left, 0));
        int y_overlap = max(0, min(bot, b->image.rows) - max(top, 0));
        int overlapArea = x_overlap * y_overlap;

        //cout << "Overlap area of " << a->id << " and " << b->id << ": " << overlapArea << endl;
        
        return overlapArea >= b->image.cols * b->image.rows * minOverlap;
    }
    
    static inline bool DecomposeHomography(const InputImageP a, const InputImageP b, const Mat &hom, Mat &r, Mat &t, bool useINRA = true) {

        if(!useINRA) {
            Mat aK3(3, 3, CV_64F);
            Mat bK3(3, 3, CV_64F);

            ScaleIntrinsicsToImage(a->intrinsics, a->image, aK3);
            ScaleIntrinsicsToImage(b->intrinsics, b->image, bK3);

            t = Mat(3, 1, CV_64F);

            From3DoubleTo4Double(bK3.inv() * hom * aK3, r);
            
            return true;
        } else {
            Mat aK3(3, 3, CV_64F);
            ScaleIntrinsicsToImage(a->intrinsics, a->image, aK3);
            
            vector<Mat> rotations(4);
            vector<Mat> translations(4);
            vector<Mat> normals(4);

			int nsols = decomposeHomographyMat(hom, aK3, rotations, translations, normals);

 			for(int i = 0; i < nsols; i++) {

 				if(!ContainsNaN(rotations[i])) {
                    From3DoubleTo4Double(rotations[i], r);
                    t = translations[i];
                    return true;
 				}
 			}

            return false;
        }
    } 
  
    /**
     * Gets the overlapping region between two images related by a perspective 
     * transfoerm. The first image is distorted, so the overlapping area matches the 
     * second image. 
     *
     * @param a Gives the transform of the first image.
     * @param b Gives the transform of the second image. 
     * @param ai The first image. 
     * @param bi the second image. 
     * @param overlapA On completion, is set to the overlapping, undistorted part of a
     * @param overlapB On completion, is set to the overlapping, undistorted part of b
     * @param buffer Additional image border to be included in the overlap, wherever possible, relative to the position of a.
     *
     * @returns The roi of the overlapping image area of b. 
     */
    static inline cv::Rect GetOverlappingRegion(const InputImageP a, const InputImageP b, const Image &ai, const Image &bi, Mat &overlapA, Mat &overlapB, int buffer, cv::Point &appliedBorder) {
        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);
        
        AssertM(ai.IsLoaded(), "Input a exists");
        AssertM(bi.IsLoaded(), "Input b exists");

        HomographyFromImages(a, b, hom, rot, ai.size());

        Mat offset = Mat::eye(3, 3, CV_64F);

        //Modify homography, so it includes buffer. 
        if(hom.at<double>(1, 2) < 0) {
            offset.at<double>(1, 2) += buffer; 
            appliedBorder.y = buffer;
        } else {
            offset.at<double>(1, 2) -= buffer; 
            appliedBorder.y = -buffer;
        }
        
        if(hom.at<double>(0, 2) < 0) {
            offset.at<double>(0, 2) += buffer; 
            appliedBorder.x = buffer;
        } else {
            offset.at<double>(0, 2) -= buffer; 
            appliedBorder.x = -buffer;
        }

        hom = offset * hom; 

        //Calculate scene corners and ROIs
        vector<Point2f> corners = GetSceneCorners(ai, hom);
        cv::Rect roi = GetInnerBoxForScene(corners);
        
        cv::Rect roib = cv::Rect(roi.x, roi.y, roi.width, roi.height);
        cv::Rect roia = cv::Rect(roi.x, roi.y, roi.width, roi.height);

        roia = roia & cv::Rect(0, 0, ai.cols, ai.rows);
        roib = roib & cv::Rect(0, 0, bi.cols, bi.rows);
            
        //Warp image, modify homography to set target. 
        offset.at<double>(0, 2) = -roia.x; 
        offset.at<double>(1, 2) = -roia.y; 
        
        hom = offset * hom;
        if(roia.width == 0 || roia.height == 0) {
            return roia;
        }
        
        warpPerspective(ai.data, overlapA, hom, roia.size(), INTER_LINEAR, BORDER_CONSTANT, 0);

        overlapB = bi.data(roib);

        return roib;
    }
 
    static inline void GetOverlappingRegion(const InputImageP a, const InputImageP b, const Image &ai, const Image &bi, Mat &overlapA, Mat &overlapB) {
        cv::Point dummy;
        GetOverlappingRegion(a, b, ai, bi, overlapA, overlapB, 0, dummy);
    }
    
    static inline void GeoToRot(double hAngle, double vAngle, Mat &res) {
        Mat hRot;
        Mat vRot;
        
        //cout << hAngle << ", " << vAngle << endl;
        
        CreateRotationY(hAngle, hRot);
        CreateRotationX(vAngle, vRot);
        
        res = hRot * vRot;
    }
    
}
#endif

