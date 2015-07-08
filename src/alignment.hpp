#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "core.hpp"
#include "support.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_ALIGNMENT_HEADER
#define OPTONAUT_ALIGNMENT_HEADER

namespace optonaut {

class MatchInfo {
public:
	bool valid;
	Mat homography;

	double hshift;
	double hrotation;
	double error;

	MatchInfo() : valid(false) {}
};

class Aligner {

private:
	Ptr<AKAZE> detector;

public:

	Aligner() : detector(AKAZE::create()) { }

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
			cout << "Debug: No matches found." << endl;
			return info;
		}

		vector<Point2f> aFeatures;
		vector<Point2f> bFeatures;

		for(size_t i = 0; i < goodMatches.size(); i++) {
			aFeatures.push_back(a->features[goodMatches[i].queryIdx].pt);
			bFeatures.push_back(b->features[goodMatches[i].trainIdx].pt);
		}

		info->homography = findHomography(aFeatures, bFeatures, CV_RANSAC);


		if(info->homography.cols != 0) {	
			info->valid = true;

			vector<Point2f> poi(5); //Points of interest
			poi[0] = Point2f(a->img.cols / 2, a->img.rows / 2); //Center
			poi[1] = Point2f(0, 0); //Left Upper
			poi[2] = Point2f(a->img.cols, 0); //Right Upper
			poi[3] = Point2f(0, a->img.rows); //Left Lower
			poi[4] = Point2f(a->img.cols, a->img.rows); //Right Lower
			vector<Point2f> poi_projected; //Points of interest

			perspectiveTransform(poi, poi_projected, info->homography);

			info->hshift = poi[0].x - poi_projected[0].x;
			double hrotUp = (poi[2].x - poi[1].x) - (poi_projected[2].x - poi_projected[1].x);
			double hrotLow = (poi[4].x - poi[3].x) - (poi_projected[4].x - poi_projected[3].x);
			
			double averageNormedSkew = (hrotUp + hrotLow) / (2 * (poi[2].x - poi[1].x)); 
			double x = 1 + averageNormedSkew;
			double l = cos(GetHorizontalFov(a->intrinsics) / 2);
			double h = 1;

			if(averageNormedSkew - l == 0) {
				info->hrotation = 0;
				info->error = 1;
			} else {
				cout << "skew " << x << endl;
				cout << "hfov " << (GetHorizontalFov(a->intrinsics) * 180 / M_PI) << endl; 
				info->hrotation = 2 * atan((sqrt(h * h * l * l + h * h - x * x) - h * l) / (h + x));
				info->error = (poi_projected[2].x - poi_projected[4].x + poi_projected[1].x - poi_projected[3].x) / 2;
 			}
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