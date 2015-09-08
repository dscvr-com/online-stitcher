#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "support.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_VISUAL_ALIGNMENT_HEADER
#define OPTONAUT_VISUAL_ALIGNMENT_HEADER

namespace optonaut {

class MatchInfo {
public:
	bool valid;
	Mat homography;

	vector<Mat> translations;
	vector<Mat> rotations;
	vector<Mat> normals;
	double error;

	MatchInfo() : valid(false) {}
};

class VisualAligner {

private:
	Ptr<AKAZE> detector;

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

    void DrawResults(const Mat &homography, const vector<DMatch> &goodMatches, const ImageP a, const ImageP b, Mat &target) {
  		drawMatches(a->img, a->features, b->img, b->features,
               goodMatches, target, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        DrawHomographyBorder(homography, a, Scalar(0, 255, 0), target);
        
        Mat estimation;
        HomographyFromKnownParameters(a, b, estimation);
        
        DrawHomographyBorder(estimation, a, Scalar(0, 0, 255), target);
    } 

public:

	VisualAligner() : detector(AKAZE::create()) { }

	void FindKeyPoints(ImageP img) {
		img->features.clear();
        Mat tmp = img->img;
        //resize(img->img, tmp, Size(img->img.cols * 0.5, img->img.rows * 0.5));

		detector->detectAndCompute(tmp, noArray(), img->features, img->descriptors);
	}

    void HomographyFromKnownParameters(const ImageP a, const ImageP b, Mat &hom) const {
        Mat R3(3, 3, CV_64F);
        Mat aK3(3, 3, CV_64F);
        Mat bK3(3, 3, CV_64F);
        
        From4DoubleTo3Double(b->originalExtrinsics.inv() * a->originalExtrinsics, R3);

        ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
        ScaleIntrinsicsToImage(b->intrinsics, b->img, bK3);

        hom = bK3 * R3 * aK3.inv();
        //hom = aK3 * aR3 * bR3.inv() * bK3.inv();
    }

	MatchInfo *FindHomography(const ImageP a, const ImageP b) {
        assert(a != NULL);
        if(a->features.empty())
			FindKeyPoints(a);
		if(b->features.empty()) 
			FindKeyPoints(b);

		MatchInfo* info = new MatchInfo();

        cout << "Visual Aligner receiving " << a->id << " and " << b->id << endl;

		BFMatcher  matcher;
		vector<vector<DMatch>> matches;
		matcher.knnMatch(a->descriptors, b->descriptors, matches, 1);

		//TODO - find "good" matches based on Dist, angle and more.
        //Also find 'certaincy' of matches, or quality of homography. 
        //
        //Simply use "expeced" by sensor to exclude groups. 
		vector<DMatch> goodMatches;
		for(size_t i = 0; i < matches.size(); i++) {
			if(matches[i].size() > 0) {
				goodMatches.push_back(matches[i][0]);
			}
		}

		if(goodMatches.size() == 0) {
			//cout << "Homography: no matches. " << endl;
			return info;
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

        cout << "inliner ratio: " << inlinerRatio << endl;
        cout << "matchCount: " << goodMatches.size() << endl;


		info->rotations = vector<Mat>(4);
		info->translations = vector<Mat>(4);

		if(info->homography.cols != 0) {	
			Mat scaledK;
			ScaleIntrinsicsToImage(a->intrinsics, a->img, scaledK);
			int nsols = decomposeHomographyMat(info->homography, scaledK, info->rotations, info->translations, info->normals);
 			//cout << "Number of solutions: " << nsols << endl;
 			for(int i = 0; i < nsols; i++) {

 				if(ContainsNaN(info->rotations[i])) {
 					info->rotations.erase(info->rotations.begin() + i);
 					info->translations.erase(info->translations.begin() + i);
 					info->normals.erase(info->normals.begin() + i);
 					nsols--;
 					i--;
 				}
 			}
			info->valid = info->rotations.size() > 0;
 		}
	
		//debug
 		
	   	if(info->homography.cols != 0) {
            Mat target; 
            DrawResults(info->homography, goodMatches, a, b, target);
		    imwrite("dbg/Homogpraphy" + ToString(a->id) + "_" + ToString(b->id) + "(C " + ToString(goodMatches.size()) + ", R " +ToString(inlinerRatio * 100) +  " ).jpg", target);
		} else {
			cout << "Debug: Homography not found." << endl;
		}
		//debug end
        
        if(inlinerRatio < 0.40 || inlinerCount < 80)
            info->valid = false;
		
		return info;
	}
};
}

#endif
