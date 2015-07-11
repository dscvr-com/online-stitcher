#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "core.hpp"
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

public:

	VisualAligner() : detector(AKAZE::create()) { }

	void FindKeyPoints(Image* img) {
		img->features.clear();

		detector->detectAndCompute(img->img, noArray(), img->features, img->descriptors);
	}

	MatchInfo *FindHomography(Image* a, Image* b) {
		if(a->features.empty())
			FindKeyPoints(a);
		if(b->features.empty()) 
			FindKeyPoints(b);

		MatchInfo* info = new MatchInfo();

		BFMatcher  matcher;
		vector<vector<DMatch>> matches;
		matcher.knnMatch(a->descriptors, b->descriptors, matches, 1);

		//TODO - find "good" matches based on Dist.
		vector<DMatch> goodMatches;
		for(size_t i = 0; i < matches.size(); i++) {
			if(matches[i].size() > 0) {
				goodMatches.push_back(matches[i][0]);
			}
		}

		if(goodMatches.size() == 0) {
			return info;
		}

		vector<Point2f> aFeatures;
		vector<Point2f> bFeatures;

		for(size_t i = 0; i < goodMatches.size(); i++) {
			aFeatures.push_back(a->features[goodMatches[i].queryIdx].pt);
			bFeatures.push_back(b->features[goodMatches[i].trainIdx].pt);
		}

		info->homography = findHomography(aFeatures, bFeatures, CV_RANSAC);

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

 				//cout << "rotation " << i << ": " << info->rotations[i] << endl;
 				//cout << "translation " << i << ": " << info->translations[i] << endl;
 				//cout << "normal " << i << ": " << info->normals[i] << endl;
 			}
			info->valid = info->rotations.size() > 0;
 		}
	
		//debug

		Mat img_matches;
  		drawMatches( a->img, a->features, b->img, b->features,
               goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		if(info->homography.cols != 0) {
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( a->img.cols, 0 );
			obj_corners[2] = cvPoint( a->img.cols, a->img.rows ); obj_corners[3] = cvPoint( 0, a->img.rows );
			std::vector<Point2f> scene_corners(4);

			perspectiveTransform(obj_corners, scene_corners, info->homography);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			Point2f offset(a->img.cols, 0);
			line( img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
			line( img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4 );
			line( img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4 );
			line( img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4 );

			//-- Show detected matches
		} else {
			cout << "Debug: Homography not found." << endl;
		}
		imwrite( "dbg/Homogpraphy" + ToString(a->id) + "_" + ToString(b->id) + ".jpg", img_matches );

		//debug end

		return info;
	}
};
}

#endif