#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

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

struct Pose {
    Mat R;
    Mat t;

    Pose(Mat R, Mat t) : R(R), t(t) { }
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

    recoverPose(c->E, c->aLocalFeatures, c->bLocalFeatures, R, t, 
           K.at<float>(0, 0), 
           Point2d(K.at<float>(0, 2), K.at<float>(1, 2)), 
           noArray());

    AssertGEM(1.0, std::abs(determinant(R)) - 0.00001, "Recovered rotation is valid.");

    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;
}

void RecoverPosePnP(
        const MatchInfoP &c,
        const Mat &K,
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
    solvePnPRansac(subcloud, subfeatures, K, R, rvec, t);
    R = Mat::eye(3, 3, CV_64F);
    Rodrigues(rvec, R);

    R = R.inv();
    t = -t;
}

inline void AddPoint(float x, float y, float z, Vec3b color, 
        const FeatureId &a, const FeatureId &b) {

    auto it = featuresToPoints.find(a); 
    size_t i;
    if(it == featuresToPoints.end()) {
        auto pt = Point3D(Point3f(x, y, z));
        pt.features.push_back(a);
        pt.features.push_back(b);
        pt.color = color;
        i = cloud.size();
        cloud.push_back(pt);
    } else {
        i = it->second;
        //AssertEQ(x, cloud[i].pt.x);
        //AssertEQ(x, cloud[i].pt.y);
        //AssertEQ(x, cloud[i].pt.z);
        cloud[i].features.push_back(a);
        cloud[i].features.push_back(b);
    }
    featuresToPoints.insert(make_pair(a, i));
    featuresToPoints.insert(make_pair(b, i));
}

void TriangulateAndAdd(
        const InputImageP &a, const InputImageP &,
        const MatchInfoP &c,
        const Mat &K, const Mat &D,
        const Mat &diffR, const Mat &diffT,
        const Mat &originR, const Mat &originT
        ) {

    Mat Rect1, Rect2, P1, P2, Q;

    stereoRectify(K, D, K, D,
           a->image.size(), 
           diffR, diffT, Rect1, Rect2, P1, P2, Q);
    
    Mat triangulated;

    triangulatePoints(P1, P2, c->aLocalFeatures, c->bLocalFeatures, triangulated);

    AssertEQ(triangulated.rows, 4);
    
    for(int i = 0; i < triangulated.cols; i++) {
        float x = triangulated.at<float>(0, i) / triangulated.at<float>(3, i);
        float y = triangulated.at<float>(1, i) / triangulated.at<float>(3, i);
        float z = triangulated.at<float>(2, i) / triangulated.at<float>(3, i);

        Mat point(Vec3d(x, y, z));

        point = originR.inv() * point;
        point = point - originT;

        FeatureId aFeature = c->aGlobalFeatures[i];
        FeatureId bFeature = c->bGlobalFeatures[i];

        AddPoint(point.at<double>(0), point.at<double>(1), point.at<double>(2), 
                a->image.data.at<Vec3b>(c->aLocalFeatures[i]), aFeature, bFeature);
    }
}

void MatchImages(const InputImageP &a, const InputImageP &b) {

    cout << "Receiving: " << a->id << " " << b->id << endl;
    AssertNEQ(a->id, b->id);
    
    MatchInfoP c = aligner.FindCorrespondence(a, b); 
    
    Mat scaledK;
    Mat K;

    ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), scaledK);
    From3DoubleTo3Float(scaledK, K);

    float distortion[] = {0, 0, 0, 0};
    Mat D = Mat(4, 1, CV_32F, distortion);

    Mat R, t;

    if(cloud.size() == 0) {
        RecoverPoseStereo(c, K, R, t);
        posesPerImage.insert(
                make_pair(
                    a->id, 
                    Pose(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F))
                    ));
    } else {
        RecoverPosePnP(c, K, R, t);
    }
    
    Pose origin = posesPerImage.at(a->id);
    Pose next = Pose(origin.R * R, origin.t + t);

    posesPerImage.insert(make_pair(b->id, next));

    TriangulateAndAdd(a, b, c, K, D, R, t, origin.R, origin.t);
}

void ShowCloud() {
    VisualDebugHook debugger;

    const float scale = 0.5;

    for(size_t i = 0; i < cloud.size(); i++) {

        Point3D p = cloud[i];

        debugger.PlaceFeature(
                p.pt.x * scale, p.pt.y * scale, p.pt.z * scale,
                p.color[2], p.color[1], p.color[0]); 
    }

    debugger.Draw();
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(
            argc, argv, false, 20, 1);

    int n = images.size();

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    for(int i = 0; i < n; i++) {
        auto img = images[i];
        img->image = Image(img->image.data);
        img->intrinsics = iPhone5Intrinsics;
       
        static const bool undistort = true;

        if(undistort) {
            float distortion[] = {0.0439, -0.0119, 0, 0};
            Mat Rect1, Rect2, P1, P2, Q, Distortion = Mat(4, 1, CV_32F, distortion);

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
