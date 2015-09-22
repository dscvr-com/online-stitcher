#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include "image.hpp"
#include "support.hpp"
#include "io.hpp"
#include "projection.hpp"
#include "drawing.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

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

    //TODO Make dependent of image size
    const double OutlinerTolerance = 155 * 155; //Chosen by fair jojo roll

    static const bool debug = true;

    //Needed for bundle adj.
    map<ImageP, size_t> imgToLocalId;
    vector<MatchesInfo> matches;
    vector<ImageFeatures> features;

    size_t GetImageId(ImageP img) {
        if(imgToLocalId.find(img) == imgToLocalId.end()) {
            imgToLocalId.insert(pair<ImageP, size_t>(img, imgToLocalId.size()));
        }
        
        size_t id = imgToLocalId[img];

        return id;
    }
public:

	PairwiseVisualAligner() : detector(AKAZE::create()) { }

	void FindKeyPoints(ImageP img) {
        size_t id = GetImageId(img);

        assert(features.size() == id);

        ImageFeatures f;

        Mat tmp = img->img;

        f.img_idx = id;
        Mat *descriptors = new Mat();
        f.img_size = img->img.size();

        detector->detectAndCompute(tmp, noArray(), f.keypoints, *descriptors);

        f.descriptors = descriptors->getUMat(ACCESS_READ);

        features.push_back(f);
    }

	MatchInfo *FindCorrespondence(const ImageP a, const ImageP b) {
        assert(a != NULL);
        assert(b != NULL);

		MatchInfo* info = new MatchInfo();
        info->valid = false;

        size_t aId = GetImageId(a);
        size_t bId = GetImageId(b);

        if(features.size() <= aId)
			FindKeyPoints(a);
		if(features.size() <= bId) 
			FindKeyPoints(b);
        
        ImageFeatures aFeatures = features[aId];
        ImageFeatures bFeatures = features[bId];

        //Estimation from image movement. 
        Mat estRot; 
        Mat estHom;
        
        HomographyFromImages(a, b, estHom, estRot);

		BFMatcher matcher;
		vector<vector<DMatch>> matches;
		matcher.knnMatch(aFeatures.descriptors, bFeatures.descriptors, matches, 1);

		vector<DMatch> goodMatches;
		for(size_t i = 0; i < matches.size(); i++) {
			if(matches[i].size() > 0) {

                std::vector<Point2f> src(1);
                std::vector<Point2f> est(1);

                src[0] = aFeatures.keypoints[matches[i][0].queryIdx].pt;
                Point2f dst = bFeatures.keypoints[matches[i][0].trainIdx].pt;

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
			return info;
		}

		vector<Point2f> aLocalFeatures;
		vector<Point2f> bLocalFeatures;

		for(size_t i = 0; i < goodMatches.size(); i++) {
			aLocalFeatures.push_back(aFeatures.keypoints[goodMatches[i].queryIdx].pt);
			bLocalFeatures.push_back(bFeatures.keypoints[goodMatches[i].trainIdx].pt);
		}

        MatchesInfo minfo;
        minfo.src_img_idx = GetImageId(a);
        minfo.dst_img_idx = GetImageId(b);
        minfo.matches = goodMatches;

		info->homography = findHomography(aLocalFeatures, bLocalFeatures, CV_RANSAC, 3, minfo.inliers_mask);

        int inlinerCount = 0;

        for(size_t i = 0; i < minfo.inliers_mask.size(); i++) {
            if(minfo.inliers_mask[i] == 1)
                inlinerCount++;
        }

        minfo.num_inliers = inlinerCount;
        minfo.confidence = 0;
        minfo.H = info->homography;

        double inlinerRatio = inlinerCount;
        inlinerRatio /= goodMatches.size();

        Mat translation;

		if(info->homography.cols != 0) {	
            info->valid = DecomposeHomography(a, b, info->homography, info->rotation, translation);
 		}

        //TODO There is some kind of REPROJ technique to check the error here. 
	
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

            if(info->valid) {
                minfo.confidence = 100;
            }
           
            //debug
            if(debug) {
                Mat target; 

                Mat aK3;
                Mat reHom, rot3(3, 3, CV_64F);
                ScaleIntrinsicsToImage(a->intrinsics, a->img, aK3);
                From4DoubleTo3Double(info->rotation, rot3);
                HomographyFromRotation(rot3, aK3, reHom);

                DrawMatchingResults(info->homography, reHom, goodMatches, a->img, aFeatures, b->img, bFeatures, target);

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
        
        this->matches.push_back(minfo);

        return info;
    }

    void RunBundleAdjustment(const vector<ImageP> &images) const {
        vector<CameraParams> cameras(images.size());

        for(auto img : images) {
            auto it = imgToLocalId.find(img);
            if(it == imgToLocalId.end())
                continue;

            size_t id = it->second;
            detail::CameraParams c; 

            c.focal = img->intrinsics.at<double>(0, 0);
            c.ppx = img->intrinsics.at<double>(0, 2);
            c.ppy = img->intrinsics.at<double>(1, 2);
            c.aspect = c.ppx / c.ppy;
            c.t = Mat(3, 1, CV_32F);
            c.R = Mat(3, 3, CV_32F);

            From4DoubleTo3Float(img->originalExtrinsics, c.R);

            cameras[id] = c;
        }

        vector<MatchesInfo> fullMatches(images.size() * images.size());

        for(auto match : matches) {
            fullMatches[match.src_img_idx * images.size() + match.dst_img_idx] = match;
        }

        auto adjuster = BundleAdjusterRay();
        //adjuster.setConfThresh(...); //?
        Mat refineMask = Mat::zeros(3, 3, CV_8U);
        adjuster.setRefinementMask(refineMask);
        adjuster.setTermCriteria(TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 100, 10e-2));
        if(!adjuster(features, fullMatches, cameras)) {
            cout << "Bundle Adjusting failed" << endl;
        }
        cout << "Bundle Adjustment finished" << endl;

        vector<Mat> rmats;
         
        for (size_t i = 0; i < cameras.size(); ++i) {
            //rmats.emplace_back(3, 3, CV_32F);
            //From3DoubleTo3Float(cameras[i].R, rmats[i]);
            rmats.push_back(cameras[i].R.clone());
            cout << "Discarding translation: " << cameras[i].t << endl;
        }
        waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);

        for(auto img : images) {
            auto it = imgToLocalId.find(img);
            if(it == imgToLocalId.end())
                continue;
            
            From3FloatTo4Double(rmats[it->second], img->adjustedExtrinsics);
        }
    }
};
}

#endif
