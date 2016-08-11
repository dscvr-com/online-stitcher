/*
 * Contains functions to do different geometric calculations, 
 * especially related to projective transforms and homographies. 
 */

#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "../io/inputImage.hpp"
#include "../common/image.hpp"
#include "../common/assert.hpp"
#include "../common/support.hpp"
#include "../common/functional.hpp"
#include "../recorder/recorderGraph.hpp"
#include "support.hpp"

#ifndef OPTONAUT_PROJECTION_HEADER
#define OPTONAUT_PROJECTION_HEADER

using namespace std;
using namespace cv;

namespace optonaut {
    /**
     * Returns the corners of an image, transformed by the given homography. 
     */
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

    static inline void PrepareMats(const Mat &r, const Mat &k, const Size &imageSize, Mat &fr, Mat &sfk) {
       Mat sk;
       From4DoubleTo3Float(r, fr);
       ScaleIntrinsicsToImage(k, imageSize, sk);
       From3DoubleTo3Float(sk, sfk);
    }

    static inline cv::Point WarpPointWarper(cv::detail::RotationWarper &warper, const cv::Mat &intrinsics, const cv::Mat &extrinsics,
            const Size &imageSize, const Point &point) {
       Mat r, k;
       PrepareMats(extrinsics, intrinsics, imageSize, r, k);
       return warper.warpPoint(point, k, r); 
    }

    static inline cv::Rect GetInnerRectangle(cv::detail::RotationWarper &warper, const cv::Mat &k, const cv::Mat &r, const Size &imageSize) {
        Point tl = warper.warpPoint(Point2f(0, 0), k, r);
        Point bl = warper.warpPoint(Point2f(0, imageSize.height), k, r);
        Point tr = warper.warpPoint(Point2f(imageSize.width, 0), k, r);
        Point br = warper.warpPoint(Point2f(imageSize.width, imageSize.height), k, r);

        Point corePos(std::max(tl.x, bl.x), std::max(tl.y, tr.y));
        Size coreSize(std::min(tr.x, br.x) - corePos.x, std::min(bl.y, br.y) - corePos.y);

        return Rect(corePos, coreSize);
    }
    
    static inline cv::Rect GetInnerRectangle(cv::detail::RotationWarper &warper, const InputImageP &img) {
        Mat r, k;
        PrepareMats(img->originalExtrinsics, img->intrinsics, img->image.size(), r, k);
        return GetInnerRectangle(warper, k, r, img->image.size());
    } 

    static inline cv::Rect GetOuterRectangle(cv::detail::RotationWarper &warper, const cv::Mat &k, const cv::Mat &r, const Size &imageSize) {
        return warper.warpRoi(imageSize, k, r);
    }
    
    static inline cv::Rect GetOuterRectangle(cv::detail::RotationWarper &warper, const InputImageP& img) {
        Mat r, k;
        PrepareMats(img->originalExtrinsics, img->intrinsics, img->image.size(), r, k);
        return warper.warpRoi(img->image.size(), k, r);
    }

    /*
     * Returns the outer bounding box of the quad defined by the given points. 
     */
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
    
    /*
     * Returns the inner box for the given scene. This function relies on the order
     * of the given points and is meant to be used together with GetSceneCorners. 
     */
    static inline cv::Rect GetInnerBoxForScene(const vector<Point2f> &c) {
        assert(c.size() == 4);

        double l = max(c[0].x, c[3].x);
        double t = max(c[0].y, c[1].y);
        double r = min(c[1].x, c[2].x);
        double b = min(c[2].y, c[3].y);

        return cv::Rect(l, t, r - l, b - t);
    }
    
    /*
     * For any given relative rotation and camera intrinsics, calculates
     * the homography between the two projection planes. 
     */
    static inline void HomographyFromRotation(const Mat &rot, const Mat &k, Mat &hom) {
        hom = k * rot * k.inv();
    }

    /*
     * For two given images, calculates rotational difference between them. 
     */
    static inline void RotationFromImages(const InputImageP a, const InputImageP b, Mat &rot) {
        rot = b->adjustedExtrinsics.inv() * a->adjustedExtrinsics;
    }

    /*
     * For two given images, calcualtes the homography between their projective planes. Output coordinates are given in pixels, 
     * in case translations happen. 
     */
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
   
    /*
     * Convenience overload for HomographyFromImages that automatically sets the scale. 
     */ 
    static inline void HomographyFromImages(const InputImageP a, const InputImageP b, Mat &hom, Mat &rot) {
        HomographyFromImages(a, b, hom, rot, a->image.size());
    }

    /*
     * Checks wether two images projected on a unit sphere are overlapping or not. 
     */
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
   
    /*
     * Decomposes a homography into rotational and translational components. 
     * Camera intrinsics are extracted from the input images. 
     */ 
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

    static inline cv::Point GetOverlappingRegionWarper(const InputImageP a, const InputImageP b, const Image &ai, 
            const Image &bi, Mat &overlapA, Mat &overlapB, int, cv::Point &) {

        AssertM(ai.IsLoaded(), "Input a exists");
        AssertM(bi.IsLoaded(), "Input b exists");

        auto warperFactory = new cv::SphericalWarper();
        auto warper = warperFactory->create(800.0f);

        Mat scaledK;
        Mat K, RA, RB;
        ScaleIntrinsicsToImage(a->intrinsics, ai, scaledK);
        From3DoubleTo3Float(scaledK, K);
        From4DoubleTo3Float(a->originalExtrinsics, RA);
        From4DoubleTo3Float(b->originalExtrinsics, RB);

        Mat awarp, bwarp;

        Rect aroi = warper->warpRoi(ai.size(), K, RA);
        Rect broi = warper->warpRoi(bi.size(), K, RB);
        
        Point aLoc= warper->warp(ai.data, K, RA, INTER_NEAREST, BORDER_CONSTANT, awarp);
        Point bLoc = warper->warp(bi.data, K, RB, INTER_NEAREST, BORDER_CONSTANT, bwarp);

        Rect roi = aroi & broi;

        // Ugh, that's ugly!
        Rect aDestRoi(roi.x - aLoc.x + 10, roi.y - aLoc.y + 10, roi.width - 20, roi.height - 20);
        Rect bDestRoi(roi.x - bLoc.x + 10, roi.y - bLoc.y + 10, roi.width - 20, roi.height - 20);

        overlapA = awarp(aDestRoi);
        overlapB = bwarp(bDestRoi);
    
        return Point(broi.x - aroi.x, broi.y - aroi.y);
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
     * @returns The location difference of b and a measured on the image plane. 
     */
    static inline cv::Point GetOverlappingRegion(const InputImageP a, const InputImageP b, const Image &ai, 
            const Image &bi, Mat &overlapA, Mat &overlapB, int buffer, cv::Point &appliedBorder) {
        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);
        
        AssertM(ai.IsLoaded(), "Input a exists");
        AssertM(bi.IsLoaded(), "Input b exists");

        HomographyFromImages(a, b, hom, rot, ai.size());

        Mat offset = Mat::eye(3, 3, CV_64F);

        Point2d locationDiff(hom.at<double>(0, 2), hom.at<double>(1, 2));

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
            return locationDiff;
        }
        
        warpPerspective(ai.data, overlapA, hom, roia.size(), INTER_LINEAR, BORDER_CONSTANT, 0);

        overlapB = bi.data(roib);

        return locationDiff;
    }

    /*
     * Convenience overload of GetOverlappingRegion.
     */ 
    static inline void GetOverlappingRegion(const InputImageP a, const InputImageP b, const Image &ai, const Image &bi, Mat &overlapA, Mat &overlapB) {
        cv::Point dummy;
        GetOverlappingRegion(a, b, ai, bi, overlapA, overlapB, 0, dummy);
    }
   
    /*
     * Converts spherical coordinates to their rotation matrix representation. 
     */ 
    static inline void GeoToRot(double hAngle, double vAngle, Mat &res) {
        Mat hRot;
        Mat vRot;
        
        //cout << hAngle << ", " << vAngle << endl;
        
        CreateRotationY(hAngle, hRot);
        CreateRotationX(vAngle, vRot);
        
        res = hRot * vRot;
    }
   
    /*
     * Extracts a cube map face from a equirectangular panorama. 
     *
     * @param optograph The equirectangular panorama to extract the cube face from.
     * @param face The resulting cubeface
     * @param faceId The index of the cubeface, from 0 to 5, inclusive
     * @param width The desired width of the result
     * @param height The height of the result
     * @param subX Relative horizontal offset for a sub cube face
     * @param subY Relative vertical offset for a sub cuve face
     * @param subW Relative witdh for a sub cube face
     * @param subH Relative height for a sub cube face
     */ 
    static inline void CreateCubeMapFace(const Mat &optograph, Mat &face,
                                  const int faceId, const int width, const int height,
                                  const float subX = 0, const float subY = 0,
                                  const float subW = 1, const float subH = 1) {
        
        static float faceTransform[6][2] =
        {
            {0, 0},
            {M_PI / 2, 0},
            {M_PI, 0},
            {-M_PI / 2, 0},
            {0, -M_PI / 2},
            {0, M_PI / 2}
        };
        
        assert(faceId >= 0 && faceId < 6);
        assert(subX >= 0 && subX + subW <= 1);
        assert(subY >= 0 && subY + subH <= 1);
        assert(subW > 0);
        assert(subH > 0);
        
        Mat in = optograph;
        float inWidth;
        float inHeight;
        float inOffsetTop;
        float inOffsetLeft;
        
        if(in.cols == in.rows) {
            // Old quadratic Optograph format, we need to cut black borders away.
            inWidth = in.cols;
            inHeight = in.rows;
            inOffsetTop = 0;
            inOffsetLeft = 0;
        } else {
            // Optimised format - fake black border
            inWidth = in.cols;
            inHeight = inWidth / 2;
            inOffsetTop = (inHeight - in.rows) / 2;
            inOffsetLeft = 0;
        }
        
        Mat mapx(height, width, CV_32F);
        Mat mapy(height, width, CV_32F);
        
        const float an = sin(M_PI / 4);
        const float ak = cos(M_PI / 4);
        
        const float ftu = faceTransform[faceId][0];
        const float ftv = faceTransform[faceId][1];
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                
                // Map texture to unit space [0, 1]
                float nx = (float)y / (float)height;
                float ny = (float)x / (float)width;
                
                // Subface
                nx *= subW;
                ny *= subH;
                
                nx += subX;
                ny += subY;
                
                // Remap to [-an, an]
                nx -= 0.5f;
                ny -= 0.5f;
                
                nx *= 2 * an;
                ny *= 2 * an;
                
                float u, v;
                
                // Project
                if(ftv == 0) {
                    // Center faces
                    u = atan2(nx, ak);
                    v = atan2(ny * cos(u), ak);
                    u += ftu;
                } else if(ftv > 0) {
                    // Bottom face
                    float d = sqrt(nx * nx + ny * ny);
                    v = M_PI / 2 - atan2(d, ak);
                    u = atan2(ny, nx);
                } else {
                    // Top face
                    float d = sqrt(nx * nx + ny * ny);
                    v = -M_PI / 2 + atan2(d, ak);
                    u = atan2(-ny, nx);
                }
                u = u / (M_PI);
                v = v / (M_PI / 2);
                
                // Warp around
                while (v < -1) {
                    v += 2;
                    u += 1;
                }
                while (v > 1) {
                    v -= 2;
                    u += 1;
                }
                
                while(u < -1) {
                    u += 2;
                }
                while(u > 1) {
                    u -= 2;
                }
                
                // Map to texture sampling space
                u = u / 2.0f + 0.5f;
                v = v / 2.0f + 0.5f;
                
                u = u * (inWidth - 1) - inOffsetLeft;
                v = v * (inHeight - 1) - inOffsetTop;
                
                // Save in map
                mapx.at<float>(x, y) = u;
                mapy.at<float>(x, y) = v; 
            }
        }
        
        if(face.cols != width || face.rows != height || face.type() != in.type()) {
            face = Mat(width, height, in.type());
        }
        remap(in, face, mapx, mapy, CV_INTER_LINEAR, 
                BORDER_CONSTANT, Scalar(0, 0, 0));
    }

    /*
     * For a vector of input images, returns a vector containing their intrinsics. 
     */
    static inline std::vector<cv::Mat> ExtractExtrinsics(
            const std::vector<InputImageP> &in) {
        return fun::map<InputImageP, cv::Mat>(in, [](const InputImageP &q) { 
                    return q->adjustedExtrinsics; 
                });
    }
    
    /*
     * For a vector of input images, returns a vector containing their extrinsics.  
     */
    static inline std::vector<cv::Mat> ExtractExtrinsics(
            const std::vector<SelectionPoint> &in) {
        return fun::map<SelectionPoint, cv::Mat>(in, [](const SelectionPoint &q) { 
                    return q.extrinsics; 
                });
    }
}
#endif

