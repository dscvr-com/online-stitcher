#include <type_traits>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
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
using namespace cv::xfeatures2d;

#ifndef OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER
#define OPTONAUT_PAIRWISE_VISUAL_ALIGNMENT_HEADER

namespace optonaut {

struct FeatureId {
    size_t imageId;
    size_t featureIndex;

    bool operator<(const FeatureId &n) const {
        return std::tie(this->imageId, this->featureIndex) <
            std::tie(n.imageId, n.featureIndex); 
    }
};

class MatchInfo {
public:
	bool valid;
	Mat E;
	Mat rotation;
	double error;
    MatchesInfo matches;

    vector<Point2f> aLocalFeatures;
    vector<FeatureId> aGlobalFeatures;
    vector<Point2f> bLocalFeatures;
    vector<FeatureId> bGlobalFeatures;

	MatchInfo() : valid(false), E(4, 4, CV_64F), rotation(4, 4, CV_64F) {}
};

typedef std::shared_ptr<MatchInfo> MatchInfoP;

template <typename Extractor, typename Matcher>
inline Matcher InstantiateMatcher() {
    return Matcher();
}

class PairwiseVisualAligner {

private:
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;

    static const bool debug = true;
    const size_t NO_CHAIN = (size_t)-1; 

    //Needed for bundle adj.
    map<InputImageP, size_t> imgToLocalId;
    vector<MatchInfoP> matches;
    vector<ImageFeatures> features;
    vector<vector<size_t>> chainRefs;
    vector<vector<FeatureId>> featureChains;

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

    Mat debugTarget;

    void AllocateDebug(const InputImageP &a, const InputImageP &b) {
        if(!debug) return;
        AssertEQ(a->image.size(), b->image.size());
        debugTarget = Mat::zeros(Size(a->image.cols * 3, a->image.rows), CV_8UC3); 
    }

    void DrawImageFeaturesDebug(const InputImageP &a, 
            const vector<KeyPoint> &keypoints1, const InputImageP &b, 
            const vector<KeyPoint> &keypoints2) {
        if(!debug) return;
            vector<char> mask; 
            vector<DMatch> matches; 
        
            a->image.data.copyTo(
                debugTarget(Rect(0, 0, a->image.cols, a->image.rows)));

            b->image.data.copyTo(
                debugTarget(Rect(a->image.cols, 0, a->image.cols, a->image.rows)));

            drawMatches(a->image.data, keypoints1, 
                    b->image.data, keypoints2, 
                    matches, debugTarget, Scalar::all(-1), Scalar(0, 0, 255), 
                    mask, DrawMatchesFlags::DRAW_RICH_KEYPOINTS | 
                    DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    void DrawFlowFieldDebug(const InputImageP &a, 
            const vector<Point2f> &matchesA, const InputImageP &b, 
            const vector<Point2f> &matchesB) {
        if(!debug) return;
        AssertEQ(a->image.size(), b->image.size());
        AssertEQ(matchesA.size(), matchesB.size());

        Point2f o(a->image.cols * 2, 0);

        // Draw homography flow field
        Mat estRot; 
        Mat estHom;
        
        HomographyFromImages(a, b, estHom, estRot);

        a->image.data.copyTo(
                debugTarget(Rect(o.x, 0, a->image.cols, a->image.rows)));

        // Green: Flow field from features
        // Blue: Flow field from homography
        // Red: Flow field LK

        
        for(size_t i = 0; i < matchesA.size(); i++) {
            std::vector<Point2f> src(1);
            std::vector<Point2f> est(1);

            src[0] = matchesA[i];

            perspectiveTransform(src, est, estHom);
            
            cv::arrowedLine(debugTarget, matchesA[i] + o, est[0] + o, 
                    Scalar(0xFF, 0x00, 0x00), 2, 8, 0, 0.1);
		}

        // Draw Flow Field LK
        /*
            for(size_t i = 0; i < matchesA.size(); i++) {
                if(status[i] == 1) {
                    cv::arrowedLine(debugTarget, matchesA[i] + o, flowB[i] + o, 
                            Scalar(0x00, 0x00, 0xFF), 2, 8, 0, 0.1);
                }
            }
        */
        
        // Draw detected matches
        for(size_t i = 0; i < matchesA.size(); i++) {
            cv::arrowedLine(debugTarget, matchesA[i] + o, matchesB[i] + o, 
                    Scalar(0x00, 0xFF, 0x00), 2, 8, 0, 0.1);
        }
        
    }

    void WriteDebug(const InputImageP &a, const InputImageP &b) {
        if(!debug) return;
        std::string filename = 
            "dbg/Homogpraphy_(" + ToString(a->id) + "_" + ToString(b->id) + ").jpg";
        imwrite(filename, debugTarget);
    }
   
public:

	PairwiseVisualAligner() : 
        detector(AKAZE::create()), 
        extractor(AKAZE::create()),
        matcher(new BFMatcher())
    { 
    }
	
    PairwiseVisualAligner(Ptr<FeatureDetector> detector,
            Ptr<DescriptorExtractor> extractor,
            Ptr<DescriptorMatcher> matcher) : 
        detector(detector), 
        extractor(extractor),
        matcher(matcher)
    { 
    }


	void FindKeyPoints(InputImageP img) {
        size_t id = GetImageId(img);

        assert(features.size() == id);

        ImageFeatures f;

        Mat tmp = img->image.data;

        f.img_idx = (int)id;
        f.img_size = img->image.data.size();
        STimer t;
        
        //int keyPtDiameter = 15;
        // DAISY
        // for(int x = keyPtDiameter; x < tmp.cols - keyPtDiameter; x += 1) {
        //      for(int y = keyPtDiameter; y < tmp.rows - keyPtDiameter; y += 1) {
        //            f.keypoints.push_back(KeyPoint(x, y, 15)); 
        //            //Last param is daisy keypoint diameter param
        //        }
        //    }

        //    detector->compute(tmp, f.keypoints, f.descriptors);
        //    t.Tick("Feature Detection and Description");
        if (detector == extractor) {
            Mat descriptors; 
            detector->detectAndCompute(tmp, noArray(), f.keypoints, descriptors);
            f.descriptors = descriptors.getUMat(ACCESS_READ);
            t.Tick("Feature Detection and Description");
        } else { 
            detector->detect(tmp, f.keypoints);
            t.Tick("Feature Detection");
            Mat descriptors; 
            extractor->compute(tmp, f.keypoints, descriptors);
            f.descriptors = descriptors.getUMat(ACCESS_READ);
            t.Tick("Feature Description");
        } 

        AllocateChainRefs(id, f.keypoints.size());

        features.push_back(f);
    }

    vector<vector<FeatureId>> &GetFeatureChains() {
        return featureChains;
    }

    ImageFeatures& GetFeaturesByImage(const InputImageP a) {
        size_t aId = GetImageId(a);
        AssertGT(features.size(), aId);
        return features[aId];
    }
    
    ImageFeatures& GetFeaturesById(size_t aId) {
        AssertGT(features.size(), aId);
        return features[aId];
    }

	MatchInfoP FindCorrespondence(const InputImageP a, const InputImageP b) {
        assert(a != NULL);
        assert(b != NULL);
        AllocateDebug(a, b);
        
		MatchInfoP info(new MatchInfo());
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
       
        STimer t;

        bool useFlowField = false;
        vector<DMatch> goodMatches;
        
        if(!useFlowField) {
            //Can we do local search here? 
            vector<vector<DMatch>> matches;

            matcher->knnMatch(aFeatures.descriptors, bFeatures.descriptors, 
                    matches, 2);
            t.Tick("Feature KNN Match");

            for(size_t i = 0; i < matches.size(); i++) {

                bool isMatch = matches[i].size() > 0;

                if(!isMatch)
                    continue;

                if(matches[i].size() > 1) {
                    bool ratioTest = 
                        matches[i][0].distance < matches[i][1].distance * 0.75;

                    if(!ratioTest)
                        continue;
                }

                std::vector<Point2f> src(1);
                std::vector<Point2f> est(1);

                src[0] = aFeatures.keypoints[matches[i][0].queryIdx].pt;
                Point2f dst = bFeatures.keypoints[matches[i][0].trainIdx].pt;

                perspectiveTransform(src, est, estHom);

                Point2f resudialDistVec = est[0] - dst;
                Point2f estDistVec = est[0] - src[0];

                // Distance between estimation and matched feature
                double resudialDist = resudialDistVec.x * resudialDistVec.x + 
                    resudialDistVec.y * resudialDistVec.y;
               
                // ESTIMATED distance between keypoints in both images.  
                double estDist = estDistVec.x * estDistVec.x + 
                    estDistVec.y * estDistVec.y;

                bool estimatedPoseCheck = estDist > resudialDist * 5;

                if(!estimatedPoseCheck)
                    continue;
                        
                goodMatches.push_back(matches[i][0]);

            }

            t.Tick("Rich Feature Outlier Rejection");

            if(goodMatches.size() == 0) {
                cout << "Homography: no matches. " << endl;
                DrawImageFeaturesDebug(a, aFeatures.keypoints, 
                        b, bFeatures.keypoints);
                WriteDebug(a, b);
                return info;
            }

            for(size_t i = 0; i < goodMatches.size(); i++) {
                info->aLocalFeatures.push_back(
                        aFeatures.keypoints[goodMatches[i].queryIdx].pt);
                info->bLocalFeatures.push_back(
                        bFeatures.keypoints[goodMatches[i].trainIdx].pt);
            }
        } else {
            Mat ga, gb; 
            
            vector<Point2f> flowA(fun::map<KeyPoint, Point2f>(aFeatures.keypoints, 
                        [](auto k) { return k.pt; }));
            vector<Point2f> flowB(flowA.size());

            cvtColor(a->image.data, ga, CV_RGB2GRAY);
            cvtColor(b->image.data, gb, CV_RGB2GRAY);

            vector<uchar> status; 
            vector<float> error;
            calcOpticalFlowPyrLK(ga, gb, flowA, flowB,
                status, error);

            t.Tick("Feature Flowfield Match");

            info->aLocalFeatures = flowA;
            info->bLocalFeatures = flowB;

            for(size_t i = 0; i < flowA.size(); i++) {
                goodMatches.emplace_back(i, i, 0);
            }

            //This is wrong if we build bigger chains!
            vector<KeyPoint> keypoints;
            KeyPoint::convert(flowB, keypoints);
            bFeatures.keypoints = keypoints;

        }
        
        MatchesInfo minfo;
        minfo.src_img_idx = (int)GetImageId(a);
        minfo.dst_img_idx = (int)GetImageId(b);
        minfo.matches = goodMatches;
    
        Mat scaledK;
        ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), scaledK);
		
        //info->E = findFundamentalMat(info->aLocalFeatures, info->bLocalFeatures, 
        //        CV_RANSAC, 1, 0.99, minfo.inliers_mask);

	    info->E = findEssentialMat(info->aLocalFeatures, info->bLocalFeatures, 
                scaledK.at<double>(0, 0), 
                Point2d(scaledK.at<double>(0, 2), scaledK.at<double>(1, 2)), 
                CV_RANSAC, 0.999, 0.5, minfo.inliers_mask);

        t.Tick("Homography Estimation");

        int inlinerCount = 0;

        vector<Point2f> aInliers; 
        vector<Point2f> bInliers; 
        
        vector<FeatureId> aGlobalFeatures; 
        vector<FeatureId> bGlobalFeatures; 

        for(size_t i = 0; i < minfo.inliers_mask.size(); i++) {
            if(minfo.inliers_mask[i] != 0) {
                aInliers.push_back(info->aLocalFeatures[i]);
                bInliers.push_back(info->bLocalFeatures[i]);

                aGlobalFeatures.push_back({aId, (size_t)goodMatches[i].queryIdx});
                bGlobalFeatures.push_back({bId, (size_t)goodMatches[i].trainIdx});
                
                AppendToFeatureChain(aId, goodMatches[i].queryIdx, 
                    bId, goodMatches[i].trainIdx);
            }
        }

        info->aGlobalFeatures = aGlobalFeatures;
        info->bGlobalFeatures = bGlobalFeatures;
        
        info->aLocalFeatures = aInliers;
        info->bLocalFeatures = bInliers;
        
        t.Tick("Chain Registration");

        minfo.num_inliers = inlinerCount;
        minfo.confidence = 0;

        double inlinerRatio = inlinerCount;
        inlinerRatio /= goodMatches.size();

        Mat translation;

        if(info->valid) {
            minfo.confidence = 100;
        }
       
        DrawImageFeaturesDebug(a, aFeatures.keypoints, b, bFeatures.keypoints);
        DrawFlowFieldDebug(a, info->aLocalFeatures, b, info->bLocalFeatures);
        WriteDebug(a, b);
        
        t.Tick("Debug Output");

        info->matches = minfo;

        this->matches.push_back(info);
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

        // Expand sparse match list to full match matrix. 
        for(auto match : matches) {
            fullMatches[match->matches.src_img_idx * images.size() + match->matches.dst_img_idx] = match->matches;
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
