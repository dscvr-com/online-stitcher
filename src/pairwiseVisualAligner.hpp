#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "support.hpp"
#include "io.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER
#define OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER

namespace optonaut {

class MatchInfo {
public:
	bool valid;
	Mat homography;
	Mat rotation;
	double error;

	MatchInfo() : valid(false), homography(4, 4, CV_64F), rotation(4, 4, CV_64F) {}
};

class PairwiseVisualAligner {

private:
	Ptr<AKAZE> detector;

    const double OutlinerTolerance = 95 * 95; //Chosen by fair jojo roll

    void DrawHomographyBorder(const Mat &homography, const ImageP left, const Scalar &color, Mat &target) {

        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); 
        obj_corners[1] = cvPoint(left->img.cols, 0);
        obj_corners[2] = cvPoint(left->img.cols, left->img.rows); 
        obj_corners[3] = cvPoint(0, left->img.rows);
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, homography);

        Point2f offset(left->img.cols, 0);
        line(target, scene_corners[0] + offset, scene_corners[1] + offset, color, 4);
        line(target, scene_corners[1] + offset, scene_corners[2] + offset, color, 4);
        line(target, scene_corners[2] + offset, scene_corners[3] + offset, color, 4);
        line(target, scene_corners[3] + offset, scene_corners[0] + offset, color, 4);
    }

    void DrawResults(const Mat &homography, const Mat &homographyFromRot, const vector<DMatch> &goodMatches, const ImageP a, const ImageP b, Mat &target) {

        //Colors: Green: Detected Homography.
        //        Red:   Estimated from Sensor.
        //        Blue:  Hmoography induced by dectected rotation. 


  		drawMatches(a->img, a->features, b->img, b->features,
               goodMatches, target, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        DrawHomographyBorder(homography, a, Scalar(0, 255, 0), target);
        DrawHomographyBorder(homographyFromRot, a, Scalar(255, 0, 0), target);
        
        Mat estimation;
        Mat rot;
        HomographyFromKnownParameters(a, b, estimation, rot);

        
        DrawHomographyBorder(estimation, a, Scalar(0, 0, 255), target);
    } 

    void StoreCorrespondence(int a, int b, const Mat &rotation) {
        std::string filename = "store/" + ToString(a) + "_" + ToString(b);
        BufferToBinFile(rotation.data, sizeof(double) * 16, filename);
    }

    bool LoadCorrespondence(int a, int b, const Mat &rotation) {
        std::string filename = "store/" + ToString(a) + "_" + ToString(b);
        if(!FileExists(filename)) {
            return false;
        }
        BufferFromBinFile(rotation.data, sizeof(double) * 16, filename);

        return true;
    }

    int mode;

    static const bool debug = true;
public:

    static const int ModeECCHom = 0;
    static const int ModeECCAffine = 1;
    static const int ModeFeatures = 2;

	PairwiseVisualAligner(int mode = ModeECCAffine) : detector(AKAZE::create()), mode(mode) { }

	void FindKeyPoints(ImageP img) {
        if(mode == ModeFeatures) {
		    img->features.clear();
            Mat tmp = img->img;
            //resize(img->img, tmp, Size(img->img.cols * 0.5, img->img.rows * 0.5));

		    detector->detectAndCompute(tmp, noArray(), img->features, img->descriptors);
	    }
    }

    void HomographyFromKnownParameters(const ImageP a, const ImageP b, Mat &hom, Mat &rot) const {
        Mat R3(3, 3, CV_64F);
        Mat aK3(3, 3, CV_64F);

        rot = b->originalExtrinsics.inv() * a->originalExtrinsics;
        
        From4DoubleTo3Double(rot, R3);

        ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);

        HomographyFromRotation(R3, aK3, hom);
    }

    void HomographyFromRotation(const Mat &rot, const Mat &k, Mat &hom) const {
        hom = k * rot * k.inv();
    }

    void CorrespondenceFromECC(const ImageP a, const ImageP b, MatchInfo* info) {
        Mat ga, gb;
        cvtColor(a->img, ga, CV_BGR2GRAY);
        cvtColor(b->img, gb, CV_BGR2GRAY); 

        const int warp = mode == ModeECCHom ? MOTION_HOMOGRAPHY : MOTION_TRANSLATION;
        Mat affine = Mat::zeros(2, 3, CV_32F);
        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);
        Mat in;

        const int iterations = 1000;
        const double eps = 1e-5;

        HomographyFromKnownParameters(a, b, info->homography, rot);
       
        if(warp == MOTION_HOMOGRAPHY) {
            From3DoubleTo3Float(info->homography, hom); 
            in = hom;
        } else {
            affine.at<float>(0, 0) = 1; 
            affine.at<float>(1, 1) = 1; 
            affine.at<float>(0, 2) = info->homography.at<double>(0, 2); 
            affine.at<float>(1, 2) = info->homography.at<double>(1, 2); 
            in = affine;
        }
        TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
        try {
            findTransformECC(ga, gb, affine, warp, termination);
        } catch (Exception ex) {
            cout << "ECC couldn't correlate" << endl;
            return;
        }
        
        if(warp == MOTION_HOMOGRAPHY) {
            hom = in;
            From3FloatTo3Double(hom, info->homography);
            info->valid = RotationFromHomography(a, b, info->homography, info->rotation);
        } else {
            affine = in;
            cout << "Found Affine: " << affine << endl;
            info->homography = Mat::eye(3, 3, CV_64F);
            info->homography.at<double>(0, 2) = affine.at<float>(0, 2); 
            info->homography.at<double>(1, 2) = affine.at<float>(1, 2); 
            info->valid = true;
        }

        //debug
        if(debug) {
            Mat target; 
            Mat aK3;
            Mat reHom, rot3(3, 3, CV_64F);
            ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
            From4DoubleTo3Double(info->rotation, rot3);
            HomographyFromRotation(rot3, aK3, reHom);

            vector<DMatch> dummy;

            DrawResults(info->homography, reHom, dummy, a, b, target);

            std::string filename;
            if(mode == MOTION_HOMOGRAPHY) {
                filename = 
                    "dbg/ecc_result" + ToString(a->id) + 
                    "_" + ToString(b->id) + ".jpg";
            } else {
                filename =  
                    "dbg/ecc_result" + ToString(a->id) + 
                    "_" + ToString(b->id) + 
                    ", x-corr: " + ToString(affine.at<float>(0, 2)) + " .jpg";
            }
            imwrite(filename, target);
        }
    }

    void CorrespondenceFromFeatures(const ImageP a, const ImageP b, MatchInfo* info) {
        cout << "Visual Aligner receiving " << a->id << " and " << b->id << endl;
        if(a->features.empty())
			FindKeyPoints(a);
		if(b->features.empty()) 
			FindKeyPoints(b);

        //Estimation from image movement. 
        Mat estRot; 
        Mat estHom;
        HomographyFromKnownParameters(a, b, estHom, estRot);

		BFMatcher matcher;
		vector<vector<DMatch>> matches;
		matcher.knnMatch(a->descriptors, b->descriptors, matches, 1);

		vector<DMatch> goodMatches;
		for(size_t i = 0; i < matches.size(); i++) {
			if(matches[i].size() > 0) {

                std::vector<Point2f> src(1);
                std::vector<Point2f> est(1);

                src[0] = a->features[matches[i][0].queryIdx].pt;
                Point2f dst = b->features[matches[i][0].trainIdx].pt;

                perspectiveTransform(src, est, estHom);

                Point2f distVec = est[0] - dst;
                double dist = distVec.x * distVec.x + distVec.y * distVec.y;

                if(dist < OutlinerTolerance) {
				    goodMatches.push_back(matches[i][0]);
                }
			}
		}

		if(goodMatches.size() == 0) {
			cout << "Homography: no matches. " << endl;
			return;
		}

		vector<Point2f> aFeatures;
		vector<Point2f> bFeatures;

		for(size_t i = 0; i < goodMatches.size(); i++) {
			aFeatures.push_back(a->features[goodMatches[i].queryIdx].pt);
			bFeatures.push_back(b->features[goodMatches[i].trainIdx].pt);
		}

        Mat mask;

		info->homography = findHomography(aFeatures, bFeatures, CV_RANSAC, 3, mask);

        int inlinerCount = 0;

        for(int i = 0; i < mask.rows; i++) {
            if(mask.at<uchar>(i) == 1)
                inlinerCount++;
        }

        double inlinerRatio = inlinerCount;
        inlinerRatio /= goodMatches.size();

		if(info->homography.cols != 0) {	
            info->valid = RotationFromHomography(a, b, info->homography, info->rotation);
 		}
	
	   	if(info->valid) {

            //Check if the homography is a translation movement (the only thing we accept)
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 2; j++) {
                    double val = info->homography.at<double>(i, j);
                    
                    if((i == j && abs(1 - val) > 0.15) || 
                       (i != j && abs(val) > 0.15)) {
                        info->valid = false;
                        break;
                    }
                }
            }
            

            //debug
            if(debug) {
                Mat target; 

                Mat aK3;
                Mat reHom, rot3(3, 3, CV_64F);
                ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
                From4DoubleTo3Double(info->rotation, rot3);
                HomographyFromRotation(rot3, aK3, reHom);

                DrawResults(info->homography, reHom, goodMatches, a, b, target);

                std::string filename = 
                    "dbg/Homogpraphy" + ToString(a->id) + "_" + ToString(b->id) + 
                    "(C " + ToString(goodMatches.size()) + 
                    ", R " + ToString(inlinerRatio * 100) +  
                    ", hom " + ToString(info->homography) + 
                    ", used " + ToString(info->valid) + 
                    ").jpg";
                imwrite(filename, target);
            }
           
		}
    }

    bool RotationFromHomography(const ImageP a, const ImageP b, const Mat &hom, Mat &r) const {

        const bool useINRA = false;

        if(!useINRA) {
            Mat aK3(3, 3, CV_64F);
            Mat bK3(3, 3, CV_64F);

            ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
            ScaleIntrinsicsToImage(b->intrinsics, b->img, bK3);

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
                    return true;
 				}
 			}

            cout << "Hom decomposition found no solutions" << endl;
            return false;
        }
    } 

	MatchInfo *FindCorrespondence(const ImageP a, const ImageP b) {
        assert(a != NULL);

		MatchInfo* info = new MatchInfo();
        info->valid = false;

        if(mode != ModeFeatures) {
            CorrespondenceFromECC(a, b, info);
        } else {
            CorrespondenceFromFeatures(a, b, info);
        }

        return info;
	}
};
}

#endif
