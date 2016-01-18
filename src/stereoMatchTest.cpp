#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

//TODO - exclude from compile target IOS
#include <opencv2/xfeatures2d.hpp>
using namespace cv::xfeatures2d;

#include "common/assert.hpp"
#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "imgproc/pairwiseVisualAligner.hpp"
#include "debug/visualDebugHook.hpp"
#include "minimal/imagePreperation.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

struct Point3D {
    Point3f pt;
    Vec3b color;
    vector<FeatureId> features; 

    Point3D(Point3f pt) : pt(pt) { }
};

void ToPoseMat(const Mat &R, const Mat &t, Mat &q) {
    Assert(MatIs(R, 3, 3, CV_64F));
    Assert(MatIs(t, 3, 1, CV_64F));
    
    q = Mat::eye(4, 4, CV_64F);

    Rect rotation = Rect(0, 0, 3, 3);
    Rect translation = Rect(3, 0, 1, 3);
    
    R.copyTo(q(rotation));
    t.copyTo(q(translation));
}

void FromPoseMat(const Mat &q, Mat &R, Mat &t) {
    Assert(MatIs(q, 4, 4, CV_64F));

    R = Mat::eye(3, 3, CV_64F);
    t = Mat::zeros(3, 1, CV_64F);
    
    Rect rotation = Rect(0, 0, 3, 3);
    Rect translation = Rect(3, 0, 1, 3);
    
    q(rotation).copyTo(R);
    q(translation).copyTo(t);
}

void InvertPose(const Mat &R, const Mat &t, Mat &iR, Mat &it) {

    Mat q;
    ToPoseMat(R, t, q);
    q = q.inv();
    FromPoseMat(q, iR, it);
}

struct Pose {
    Mat R;
    Mat t;

    Pose(Mat R, Mat t) : R(R), t(t) { }

    Pose() : R(Mat::eye(3, 3, CV_64F)), t(Mat::zeros(3, 1, CV_64F)) { }
};

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner(
    SURF::create().dynamicCast<FeatureDetector>(), 
    SURF::create().dynamicCast<DescriptorExtractor>(), 
    new BFMatcher()); 

vector<Point3D> cloud;
std::map<FeatureId, size_t> featuresToPoints;
std::map<int, Pose> posesPerImage;

void RecoverPoseStereo(
        const MatchInfoP &c,
        const Mat &K,
        Mat &R, Mat &t) {

    AssertGT(c->aLocalFeatures.size(), (size_t)0);    
    AssertGT(c->bLocalFeatures.size(), (size_t)0);    

    double focal = K.at<double>(0, 0);
    Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));

    recoverPose(c->E, c->aLocalFeatures, c->bLocalFeatures, R, t, 
           focal, pp,
           noArray());

    AssertGEM(1.0, std::abs(determinant(R)) - 0.00001, 
            "Recovered rotation is valid.");

    cout << "### Recovered Stereo" << endl;
    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;
    cout << "f: " << focal << endl;
    cout << "PP: " << pp << endl;
    
    //InvertPose(R, t, R, t);
}

void RecoverPosePnP(
        const MatchInfoP &c,
        const Mat &K,
        const Mat &D,
        Mat &R, Mat &t) {

    vector<Point3f> subcloud;
    vector<Point2f> subfeatures;

    for(size_t i = 0; i < c->aGlobalFeatures.size(); i++) {
        auto it = featuresToPoints.find(c->aGlobalFeatures[i]);

        if (it != featuresToPoints.end()) {
            subcloud.push_back(cloud[it->second].pt);
            subfeatures.push_back(c->aLocalFeatures[i]);
        }
    }

    Mat rvec;

    //TODO - param tuning. Can also use original guess. 
    solvePnPRansac(subcloud, subfeatures, K, D, rvec, t, false, 2000, 1, 0.999);
    R = Mat::eye(3, 3, CV_64F);
    Rodrigues(rvec, R);

    cout << "### Recovered PNP" << endl;
    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;

    //InvertPose(R, t, R, t);
}

inline void AddNewPoint(float x, float y, float z, Vec3b color, 
        const FeatureId &a, const FeatureId &b) {

    auto pt = Point3D(Point3f(x, y, z));
    pt.features.push_back(a);
    pt.features.push_back(b);
    pt.color = color;
    size_t pointId = cloud.size();
    cloud.push_back(pt);
    featuresToPoints.insert(make_pair(a, pointId));
    featuresToPoints.insert(make_pair(b, pointId));
}

inline void AddExistingPoint(size_t pointId, const FeatureId &a, const FeatureId &b) {
    cloud[pointId].features.push_back(a);
    cloud[pointId].features.push_back(b);
    featuresToPoints.insert(make_pair(a, pointId));
    featuresToPoints.insert(make_pair(b, pointId));
}

void TriangulateAndAdd(
        const InputImageP &a, const InputImageP &,
        const MatchInfoP &c,
        const Mat &K, const Mat &D,
        Mat &diffR, Mat &diffT,
        const Mat &originR, const Mat &originT
        ) {

    Mat Rect1, Rect2, P1, P2, Q;

    stereoRectify(K, D, K, D,
           a->image.size(), 
           diffR, diffT, Rect1, Rect2, P1, P2, Q);
    
    Mat triangulated;

    triangulatePoints(P1, P2, c->aLocalFeatures, c->bLocalFeatures, triangulated);
    
    cout << "### Triangulating with relative origin: " << endl;
    cout << "t: " << originT.t() << endl;
    cout << "R: " << originR << endl;

    AssertEQ(triangulated.rows, 4);

    vector<Point3f> originalPoints;
    vector<Point3f> matches;
    vector<Vec3d> newPoints(triangulated.cols);
    vector<int> pointIds(triangulated.cols);
    
    for(int i = 0; i < triangulated.cols; i++) {
        float w = triangulated.at<float>(3, i);
        float x = triangulated.at<float>(0, i) / w;
        float y = triangulated.at<float>(1, i) / w; 
        float z = triangulated.at<float>(2, i) / w; 

        Mat point(Vec3d(x, y, z));
        
        point = originR * point;
        point = point + originT;
            
        newPoints[i] = Vec3d(point);
    
        FeatureId aFeature = c->aGlobalFeatures[i];
        auto it = featuresToPoints.find(aFeature); 
        if(it == featuresToPoints.end()) {
            pointIds[i] = -1;
        } else {
            pointIds[i] = (int)(it->second);
            originalPoints.push_back(cloud[it->second].pt);
            matches.emplace_back(point);
        }
    }

    Mat proj = Mat::eye(3, 4, CV_64F);

    if(matches.size() > 0) {
    //    std::vector<uchar> inliers(matches.size());
    //    AssertEQ(originalPoints.size(), matches.size());
    //
    //    estimateAffine3D(matches, originalPoints, proj, inliers, 1, 0.99);
    }

    Mat nR = proj(Rect(0, 0, 3, 3));
    Mat nt = proj(Rect(3, 0, 1, 3));

    for(size_t i = 0; i < newPoints.size(); i++) {
        FeatureId aFeature = c->aGlobalFeatures[i];
        FeatureId bFeature = c->bGlobalFeatures[i];

        Mat point = nR * Mat(newPoints[i]) + nt;
        int pointId = pointIds[i];

        if(pointId == -1) {
            AddNewPoint(point.at<double>(0), point.at<double>(1), point.at<double>(2),
                a->image.data.at<Vec3b>(c->aLocalFeatures[i]), aFeature, bFeature);
        } else {
            AddExistingPoint(pointId, aFeature, bFeature);
        }

    }
}

void MatchImages(const InputImageP &a, const InputImageP &b) {

    cout << "Receiving: " << a->id << " " << b->id << endl;
    AssertNEQ(a->id, b->id);
    
    MatchInfoP c = aligner.FindCorrespondence(a, b); 
    
    Mat scaledK;

    ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), scaledK);

    double distortion[] = {0, 0, 0, 0};
    Mat D = Mat(4, 1, CV_64F, distortion);

    Mat R, t;
    Pose origin; 

    if(cloud.size() == 0) {
        RecoverPoseStereo(c, scaledK, R, t);
        origin = Pose(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F));
        posesPerImage.insert(make_pair(a->id, origin));
    } else {
        origin = posesPerImage.at(a->id);
        RecoverPosePnP(c, scaledK, D, R, t);
    }

    TriangulateAndAdd(a, b, c, scaledK, D, R, t, origin.R, origin.t);
    
    Pose next = Pose(R, t);

    posesPerImage.insert(make_pair(b->id, next));
}

void ShowCloud() {
    VisualDebugHook debugger;

    const float scale = 1;

    for(size_t i = 0; i < cloud.size(); i++) {

        Point3D p = cloud[i];

        debugger.PlaceFeature(
                p.pt.x * scale, p.pt.y * scale, p.pt.z * scale,
                p.color[2], p.color[1], p.color[0]); 
    }

    int c = 0;

    for(auto pair : posesPerImage) {
        Mat &t = pair.second.t;

        Assert(MatIs(t, 3, 1, CV_64F));
        Assert(MatIs(pair.second.R, 3, 3, CV_64F));

        Mat R;

        From3DoubleTo4Double(pair.second.R, R);
       
        debugger.RegisterCamera(
                R, t.at<double>(0), t.at<double>(1), t.at<double>(2), 
                255 * ((float)c / (float)(posesPerImage.size() - 1)));

        c++;
    }

    debugger.Draw();
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(
            argc, argv, false, 20, 10);

    int n = images.size();

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    for(int i = 0; i < n; i++) {
        auto img = images[i];
        //pyrDown(img->image.data, img->image.data);
        img->image = Image(img->image.data);
        img->intrinsics = iPhone5Intrinsics;
       
        static const bool undistort = true;

        if(undistort) {
            double distortion[] = {0.0184, 0.1045, 0, 0};
            Mat Rect1, Rect2, P1, P2, Q, Distortion = Mat(4, 1, CV_64F, distortion);

            Mat scaledK;
            ScaleIntrinsicsToImage(img->intrinsics, img->image.size(), scaledK);
            Mat undistorted;
            cv::undistort(img->image.data, undistorted, scaledK, Distortion);
            img->image = Image(undistorted);
        }

        cout << "Pushing " << img->id << endl;
        combiner.Push(img);
    }

    //combiner.Flush();

    ShowCloud();

    return 0;
}
