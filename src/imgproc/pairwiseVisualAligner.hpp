#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include "../common/drawing.hpp"
#include "../common/static_timer.hpp"
#include "../common/image.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

#ifndef OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER
#define OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER

namespace optonaut {

class MatchInfo {
public:
	bool valid;
	Mat F;
	Mat rotation;
	double error;

	MatchInfo() : valid(false), F(4, 4, CV_64F), rotation(4, 4, CV_64F) {}

};

class FeatureChainInfo {
public:
    size_t imageId;
    size_t featureIndex;
};

class PairwiseVisualAligner {

private:
	Ptr<AKAZE> detector;

    //TODO Make dependent of image size
    const double OutlinerTolerance = 155 * 155; 

    static const bool debug = true;
    const size_t NO_CHAIN = (size_t)-1; 

    //Needed for bundle adj.
    map<InputImageP, size_t> imgToLocalId;
    vector<MatchesInfo> matches;
    vector<ImageFeatures> features;
    vector<vector<size_t>> chainRefs;
    vector<vector<FeatureChainInfo>> featureChains;

    size_t CreateNewChain() {
        size_t id = featureChains.size();
        featureChains.push_back({});
        return id;
    }

    void AllocateChainRefs(size_t imageId, size_t featureCount) {
        assert(chainRefs.size() == imageId);
        chainRefs.emplace_back(featureCount);
        std::fill(chainRefs[imageId].begin(), chainRefs[imageId].end(), NO_CHAIN);
    }

    void AppendToFeatureChain(size_t trainImage, size_t trainFeature, size_t queryImage, size_t queryFeature) {
        size_t chain = chainRefs[trainImage][trainFeature]; 
        if(chain == NO_CHAIN) {
            chain = CreateNewChain();
            featureChains[chain].push_back({trainImage, trainFeature});
            chainRefs[trainImage][trainFeature] = chain;
        }

        chainRefs[queryImage][queryFeature] = chain;
        featureChains[chain].push_back({queryImage, queryFeature});
    } 

    size_t GetImageId(InputImageP img) {
        if(imgToLocalId.find(img) == imgToLocalId.end()) {
            imgToLocalId.insert(pair<InputImageP, size_t>(img, imgToLocalId.size()));
        }
        
        size_t id = imgToLocalId[img];

        return id;
    }

    struct CompareById {
        const size_t v;
        CompareById(const size_t& v) : v(v) {}
        bool operator()(const std::map<InputImageP, size_t>::value_type& p) {
            return p.second == v;
        }
    };
public:

	PairwiseVisualAligner() : detector(AKAZE::create()) { }

	void FindKeyPoints(InputImageP img) {
        size_t id = GetImageId(img);

        assert(features.size() == id);

        ImageFeatures f;

        Mat tmp = img->image.data;

        f.img_idx = (int)id;
        Mat *descriptors = new Mat();
        f.img_size = img->image.data.size();

        detector->detectAndCompute(tmp, noArray(), f.keypoints, *descriptors);

        f.descriptors = descriptors->getUMat(ACCESS_READ);
        AllocateChainRefs(id, f.keypoints.size());

        features.push_back(f);
    }

    const vector<vector<FeatureChainInfo>> &GetFeatureChains() {
        return featureChains;
    }

    const vector<ImageFeatures> &GetFeatures() {
        return features;
    }
    
    InputImageP GetImageById(const size_t id) {
        return std::find_if(imgToLocalId.begin(), 
                imgToLocalId.end(), 
                CompareById(id))->first;
    }

	MatchInfo *FindCorrespondence(const InputImageP a, const InputImageP b) {
        assert(a != NULL);
        assert(b != NULL);
        
        STimer imageMatchingTimer;

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
       
        imageMatchingTimer.Tick("Feature Detection"); 

        //Can we do local search here? 
		BFMatcher matcher;
		vector<vector<DMatch>> matches;
		matcher.knnMatch(aFeatures.descriptors, bFeatures.descriptors, matches, 2);

		vector<DMatch> goodMatches;
		for(size_t i = 0; i < matches.size(); i++) {
			if(matches[i].size() > 1 && matches[i][0].distance < matches[i][1].distance * 0.75 ) {
                std::vector<Point2f> src(1);
                std::vector<Point2f> est(1);

                src[0] = aFeatures.keypoints[matches[i][0].queryIdx].pt;
                Point2f dst = bFeatures.keypoints[matches[i][0].trainIdx].pt;

                perspectiveTransform(src, est, estHom);

                Point2f distVec = est[0] - dst;
                double dist = distVec.x * distVec.x + distVec.y * distVec.y;

                if(dist < OutlinerTolerance) {
				    goodMatches.push_back(matches[i][0]);
                    AppendToFeatureChain(aId, matches[i][0].queryIdx, 
                            bId, matches[i][0].trainIdx);
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
        minfo.src_img_idx = (int)GetImageId(a);
        minfo.dst_img_idx = (int)GetImageId(b);
        minfo.matches = goodMatches;
        imageMatchingTimer.Tick("Feature Maching");

		info->F = findFundamentalMat(aLocalFeatures, bLocalFeatures, CV_RANSAC, 3, 0.99, minfo.inliers_mask);

        imageMatchingTimer.Tick("Homography Estimation");

        int inlinerCount = 0;

        for(size_t i = 0; i < minfo.inliers_mask.size(); i++) {
            if(minfo.inliers_mask[i] == 1)
                inlinerCount++;
        }

        minfo.num_inliers = inlinerCount;
        minfo.confidence = 0;
        minfo.H = info->F;

        double inlinerRatio = inlinerCount;
        inlinerRatio /= goodMatches.size();

        Mat translation;

        if(info->valid) {
            minfo.confidence = 100;
        }
       
        //debug
        if(debug) {
            Mat target; 

            vector<char> mask; //(minfo.inliers_mask.begin(), minfo.inliers_mask.end());
            vector<DMatch> matches; // = minfo.matches;

            drawMatches(a->image.data, features[GetImageId(a)].keypoints, b->image.data, features[GetImageId(b)].keypoints, matches, target, Scalar::all(-1), Scalar(0, 0, 255), mask, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string filename = 
                "dbg/Homogpraphy" + ToString(a->id) + "_" + ToString(b->id) + 
                "(C " + ToString(goodMatches.size()) + 
                ", R " + ToString(inlinerRatio * 100) +  
                ", used " + ToString(info->valid) + 
                ").jpg";
            imwrite(filename, target);
        }
        
        imageMatchingTimer.Tick("Debug Output");

        this->matches.push_back(minfo);
        imageMatchingTimer.Tick("Result Storeage");

        return info;
    }

    void RunBundleAdjustment(const vector<InputImageP> &images) const {
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
