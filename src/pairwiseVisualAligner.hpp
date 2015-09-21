#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include "image.hpp"
#include "support.hpp"
#include "io.hpp"

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

    void DrawHomographyBorder(const Mat &homography, const Mat &left, const Scalar &color, Mat &target) {
        std::vector<Point2f> scene_corners = GetSceneCorners(left, homography);

        Point2f offset(left.cols, 0);

        for(size_t i = 0; i < scene_corners.size(); i++) {
            scene_corners[i] += offset;
        }

        DrawPoly(target, scene_corners, color);
    }

    void DrawPoly(const Mat &target, const vector<Point2f> &corners, const Scalar color = Scalar(255, 0, 0)) {
        
        Point2f last = corners.back();

        for(auto point : corners) {
            line(target, last, point, color, 4);
            last = point;
        }
    }

    void DrawBox(const Mat &target, const Rect &roi, const Scalar color = Scalar(255, 0, 0)) {
        std::vector<Point2f> corners;
        corners.emplace_back(roi.x, roi.y);
        corners.emplace_back(roi.x, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y + roi.height);
        corners.emplace_back(roi.x + roi.width, roi.y);

        DrawPoly(target, corners, color);
    }

    vector<Point2f> GetSceneCorners(const Mat &img, const Mat &homography) {
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); 
        obj_corners[1] = cvPoint(img.cols, 0);
        obj_corners[2] = cvPoint(img.cols, img.rows); 
        obj_corners[3] = cvPoint(0, img.rows);
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, homography);

        return scene_corners;
    }

    Rect GetInnerBoxForScene(const vector<Point2f> &c) {
        assert(c.size() == 4);

        double l = max(c[0].x, c[3].x);
        double t = max(c[0].y, c[1].y);
        double r = min(c[1].x, c[2].x);
        double b = min(c[2].y, c[3].y);

        return Rect(l, t, r - l, b - t);
    }

    void DrawResults(const Mat &homography, const Mat &homographyFromRot, const vector<DMatch> &goodMatches, const Mat &a, const ImageFeatures &aFeatures, const Mat &b, const ImageFeatures &bFeatures, Mat &target) {

        //Colors: Green: Detected Homography.
        //        Red:   Estimated from Sensor.
        //        Blue:  Hmoography induced by dectected rotation. 

  		drawMatches(a, aFeatures.keypoints, b, bFeatures.keypoints,
               goodMatches, target, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        DrawHomographyBorder(homography, a, Scalar(0, 255, 0), target);
        DrawHomographyBorder(homographyFromRot, a, Scalar(255, 0, 0), target);
        
        //Mat estimation;
        //Mat rot;
        //HomographyFromKnownParameters(a, b, estimation, rot);
        //DrawHomographyBorder(estimation, a, Scalar(0, 0, 255), target);
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

    static const int ModeECCHom = 0;
    static const int ModeECCAffine = 1;
    static const int ModeFeatures = 2;

	PairwiseVisualAligner(int mode = ModeFeatures) : detector(AKAZE::create()), mode(mode) { }

	void FindKeyPoints(ImageP img) {
        if(mode == ModeFeatures) {
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
    }

    bool AreOverlapping(const ImageP a, const ImageP b, double minOverlap = 0.1) {
        Mat hom, rot;

        HomographyFromKnownParameters(a, b, hom, rot);

        std::vector<Point2f> corners = GetSceneCorners(a->img, hom); 

        int top = min(corners[0].x, corners[1].x); 
        int bot = max(corners[2].x, corners[3].x); 
        int left = min(corners[0].y, corners[3].y); 
        int right = max(corners[1].y, corners[2].y); 

        int x_overlap = max(0, min(right, b->img.cols) - max(left, 0));
        int y_overlap = max(0, min(bot, b->img.rows) - max(top, 0));
        int overlapArea = x_overlap * y_overlap;

        cout << "Overlap area of " << a->id << " and " << b->id << ": " << overlapArea << endl;
        
        return overlapArea >= b->img.cols * b->img.rows * minOverlap;
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

        //GetGradient(ga, ga);
        //GetGradient(gb, gb);
        
        Mat hom(3, 3, CV_64F);
        Mat rot(4, 4, CV_64F);

        HomographyFromKnownParameters(a, b, hom, rot);
        
        //double dx = hom.at<double>(0, 2);
        //double dy = hom.at<double>(1, 2);

        Mat wa(ga.rows, ga.cols, CV_64F);
        warpPerspective(ga, wa, hom, wa.size(), INTER_LINEAR, BORDER_CONSTANT, 0);
       
        ga = wa;
        //Cut images, set homography to id.
        
        vector<Point2f> corners = GetSceneCorners(ga, hom);
        Rect roi = GetInnerBoxForScene(corners);
        //DrawBox(ga, roi, Scalar(0x70));
        //DrawBox(gb, roi, Scalar(0x70));
        roi = roi & Rect(0, 0, ga.cols, ga.rows);
        //DrawPoly(ga, corners, Scalar(0xc0));
        //DrawPoly(gb, corners, Scalar(0xc0));
        //DrawBox(ga, roi, Scalar(255));
        //DrawBox(gb, roi, Scalar(255));
        hom = Mat::eye(3, 3, CV_32F);
        ga = ga(roi);
        gb = gb(roi);

        //If those asserts fire, we've fed the aligner two non-overlapping 
        //images probably. SHAME!
        if(roi.width < 1 || roi.height < 1) {
            return;
        }

        //reduce(ga, ga, 0, CV_REDUCE_AVG);
        //reduce(gb, gb, 0, CV_REDUCE_AVG);

        //hom.at<double>(0, 2) = dx;
        //
        //hom.at<double>(1, 2) = dy;

        const int warp = mode == ModeECCHom ? MOTION_HOMOGRAPHY : MOTION_TRANSLATION;
        Mat affine = Mat::zeros(2, 3, CV_32F);
        Mat in;

        const int iterations = 1000;
        const double eps = 1e-5;

        if(warp == MOTION_HOMOGRAPHY) {
            From3DoubleTo3Float(info->homography, hom); 
            in = hom;
        } else {
            affine.at<float>(0, 0) = 1; 
            affine.at<float>(1, 1) = 1; 
            affine.at<float>(0, 2) = 0; //info->homography.at<double>(0, 2); 
            affine.at<float>(1, 2) = 0; //info->homography.at<double>(1, 2); 
            in = affine;
        }
        TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
        try {
            findTransformECC(ga, gb, in, warp, termination);
        
            if(warp == MOTION_HOMOGRAPHY) {
                hom = in;
                From3FloatTo3Double(hom, info->homography);
                info->valid = RotationFromHomography(a, b, info->homography, info->rotation);
            } else {
                affine = in;
                cout << "Found Affine: " << affine << endl;
                From3FloatTo3Double(hom, info->homography);
                info->homography.at<double>(0, 2) = affine.at<float>(0, 2); 
                info->homography.at<double>(1, 2) = affine.at<float>(1, 2); 
                info->valid = true;
            }
        } catch (Exception ex) {
            cout << "ECC couldn't correlate" << endl;
            info->homography = hom;
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

            DrawResults(info->homography, reHom, dummy, ga, ImageFeatures(), gb, ImageFeatures(), target);

            std::string filename;
            if(mode == MOTION_HOMOGRAPHY) {
                filename = 
                    "dbg/ecc_result" + ToString(a->id) + 
                    "_" + ToString(b->id) + ".jpg";
            } else {
                filename =  
                    "dbg/ecc_result" + ToString(a->id) + 
                    "_" + ToString(b->id) + 
                    "_x-corr " + ToString(affine.at<float>(0, 2)) + " .jpg";
            }
            imwrite(filename, target);
        }
    }

    void CorrespondenceFromFeatures(const ImageP a, const ImageP b, MatchInfo* info) {
        cout << "Visual Aligner receiving " << a->id << " and " << b->id << endl;
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
        
        HomographyFromKnownParameters(a, b, estHom, estRot);

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
			return;
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

                DrawResults(info->homography, reHom, goodMatches, a->img, aFeatures, b->img, bFeatures, target);

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
    }

    bool RotationFromHomography(const ImageP a, const ImageP b, const Mat &hom, Mat &r) const {

        const bool useINRA = true;

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
        assert(b != NULL);

        cout << "Visual aligner receiving " << a->id << " and " << b->id << endl;

		MatchInfo* info = new MatchInfo();
        info->valid = false;

        if(mode != ModeFeatures) {
            CorrespondenceFromECC(a, b, info);
        } else {
            CorrespondenceFromFeatures(a, b, info);
        }

        return info;
	}

    void RunBundleAdjustment(const vector<ImageP> &images) const {
        assert(mode == ModeFeatures);
    
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
