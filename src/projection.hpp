#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_PROJECTION_HEADER
#define OPTONAUT_PROJECTION_HEADER

using namespace std;
using namespace cv;

namespace optonaut {
    static inline vector<Point2f> GetSceneCorners(const Mat &img, const Mat &homography) {
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); 
        obj_corners[1] = cvPoint(img.cols, 0);
        obj_corners[2] = cvPoint(img.cols, img.rows); 
        obj_corners[3] = cvPoint(0, img.rows);
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, homography);

        return scene_corners;
    }

    static inline Rect GetBoundingBox(const vector<Point2f> &points) {
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

        return Rect(l, t, r - l, b - t);
    }
    
    static inline Rect GetInnerBoxForScene(const vector<Point2f> &c) {
        assert(c.size() == 4);

        double l = max(c[0].x, c[3].x);
        double t = max(c[0].y, c[1].y);
        double r = min(c[1].x, c[2].x);
        double b = min(c[2].y, c[3].y);

        return Rect(l, t, r - l, b - t);
    }
    
    static inline void HomographyFromRotation(const Mat &rot, const Mat &k, Mat &hom) {
        hom = k * rot * k.inv();
    }

    static inline void RotationFromImages(const ImageP a, const ImageP b, Mat &rot) {
        rot = b->originalExtrinsics.inv() * a->originalExtrinsics;
    }

    static inline void HomographyFromImages(const ImageP a, const ImageP b, Mat &hom, Mat &rot) {
        Mat R3(3, 3, CV_64F);
        Mat aK3(3, 3, CV_64F);

        RotationFromImages(a, b, rot);
        
        From4DoubleTo3Double(rot, R3);

        ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);

        HomographyFromRotation(R3, aK3, hom);
    }
    
    static inline bool ImagesAreOverlapping(const ImageP a, const ImageP b, double minOverlap = 0.1) {
        Mat hom, rot;

        HomographyFromImages(a, b, hom, rot);

        std::vector<Point2f> corners = GetSceneCorners(a->img, hom); 

        int top = min(corners[0].x, corners[1].x); 
        int bot = max(corners[2].x, corners[3].x); 
        int left = min(corners[0].y, corners[3].y); 
        int right = max(corners[1].y, corners[2].y); 

        int x_overlap = max(0, min(right, b->img.cols) - max(left, 0));
        int y_overlap = max(0, min(bot, b->img.rows) - max(top, 0));
        int overlapArea = x_overlap * y_overlap;

        //cout << "Overlap area of " << a->id << " and " << b->id << ": " << overlapArea << endl;
        
        return overlapArea >= b->img.cols * b->img.rows * minOverlap;
    }
    
    static inline bool DecomposeHomography(const ImageP a, const ImageP b, const Mat &hom, Mat &r, Mat &t, bool useINRA = true) {

        if(!useINRA) {
            Mat aK3(3, 3, CV_64F);
            Mat bK3(3, 3, CV_64F);

            ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
            ScaleIntrinsicsToImage(b->intrinsics, b->img, bK3);

            t = Mat(3, 1, CV_64F);

            From3DoubleTo4Double(bK3.inv() * hom * aK3, r);
            
            return true;
        } else {
            Mat aK3(3, 3, CV_64F);
            ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
            
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

}

#endif

