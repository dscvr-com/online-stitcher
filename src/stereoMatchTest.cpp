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

static const bool showTriangulation = true;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner(
    SURF::create().dynamicCast<FeatureDetector>(), 
    SURF::create().dynamicCast<DescriptorExtractor>(), 
    new BFMatcher()); 

void MatchImages(const InputImageP &a, const InputImageP &b) {

    cout << "Receiving: " << a->id << " " << b->id << endl;
    AssertNEQ(a->id, b->id);
    
    auto c = aligner.FindCorrespondence(a, b); 
    
    if(!showTriangulation) {
        return;
    }

    Mat R, t;

    Mat scaledK;
    Mat scaledKAsFloat;
    ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), scaledK);
    From3DoubleTo3Float(scaledK, scaledKAsFloat);

    AssertGT(c->aLocalFeatures.size(), (size_t)0);    
    AssertGT(c->bLocalFeatures.size(), (size_t)0);    

    recoverPose(c->E, c->aLocalFeatures, c->bLocalFeatures, R, t, 
           scaledK.at<double>(0, 0), 
           Point2d(scaledK.at<double>(0, 2), scaledK.at<double>(1, 2)), 
           noArray());

    AssertGEM(1.0, std::abs(determinant(R)), "Recovered rotation is valid.");

    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;

    float distortion[] = {0, 0, 0, 0};

    //float distortion[] = {0.0439, -0.0119, 0, 0};
    Mat Rect1, Rect2, P1, P2, Q, Distortion = Mat(4, 1, CV_32F, distortion);

    stereoRectify(scaledKAsFloat, Distortion, scaledKAsFloat, Distortion,
           a->image.size(), 
           R, t, Rect1, Rect2, P1, P2, Q);
    
    //fisheye::undistortPoints(c->aLocalFeatures, c->aLocalFeatures, scaledKAsFloat, 
    //        Distortion);
    
    //fisheye::undistortPoints(c->bLocalFeatures, c->bLocalFeatures, scaledKAsFloat, 
    //        Distortion);

    Mat triangulated;

    triangulatePoints(P1, P2, c->aLocalFeatures, c->bLocalFeatures, triangulated);

    VisualDebugHook debugger;

    float ax = 0;
    float az = 0;
    float ay = 0;

    const float scale = 0.5;
    const float depth = 1;
    
    for(int i = 0; i < triangulated.cols; i++) {
        ax += triangulated.at<float>(0, i) / triangulated.at<float>(3, i);
        ay += triangulated.at<float>(1, i) / triangulated.at<float>(3, i);
        az += triangulated.at<float>(2, i) / triangulated.at<float>(3, i);
    }

    ax /= triangulated.cols;
    ay /= triangulated.cols;
    az /= triangulated.cols;

    AssertEQ(triangulated.rows, 4);

    for(int i = 0; i < triangulated.cols; i++) {
        cout << triangulated.at<float>(0, i) / triangulated.at<float>(3, i) << ", " <<
                triangulated.at<float>(1, i) / triangulated.at<float>(3, i) << ", " <<
                triangulated.at<float>(2, i) / triangulated.at<float>(3, i) << ", " <<
                triangulated.at<float>(3, i) << endl; 

        debugger.PlaceFeature(
                (triangulated.at<float>(0, i) / 
                 triangulated.at<float>(3, i) - ax) * scale,
                (triangulated.at<float>(1, i) / 
                 triangulated.at<float>(3, i) - ay) * scale,
                (triangulated.at<float>(2, i) / 
                 triangulated.at<float>(3, i) - az) * scale * depth,
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[2],
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[1],
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[0]);
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
        if(i % 2 == 1)
            combiner.Flush();
    }

    return 0;
}
