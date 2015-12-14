#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "imgproc/pairwiseVisualAligner.hpp"
#include "debug/visualDebugHook.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner;

void MatchImages(const InputImageP &a, const InputImageP &b) {
    auto c = aligner.FindCorrespondence(a, b); 

        cout << "Receiving: " << a->id << " " << b->id << endl;
    AssertNEQ(a->id, b->id);
    AssertNEQ(a->image.source, b->image.source);

    Mat R, t;

    Mat scaledK;
    Mat scaledKAsFloat;
    ScaleIntrinsicsToImage(a->intrinsics, a->image.size(), scaledK);
    From3DoubleTo3Float(scaledK, scaledKAsFloat);

    AssertEQ(c->aLocalFeatures.size(), c->matches.inliers_mask.size());    
    AssertEQ(c->bLocalFeatures.size(), c->matches.inliers_mask.size());    

    Mat wrappedMask(c->matches.inliers_mask.size(), 1, CV_8U, &(c->matches.inliers_mask[0]));

    recoverPose(c->E, c->aLocalFeatures, c->bLocalFeatures, R, t, 
           scaledK.at<double>(0, 0), 
           Point2d(scaledK.at<double>(0, 2), scaledK.at<double>(1, 2)), 
           wrappedMask);

    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;

    Mat Rect1, Rect2, P1, P2, Q, Distortion = Mat::zeros(4, 1, CV_32F);

    stereoRectify(scaledKAsFloat, Distortion, scaledKAsFloat, Distortion,
           a->image.size(), 
           R, t, Rect1, Rect2, P1, P2, Q);

    Mat triangulated;

    triangulatePoints(P1, P2, c->aLocalFeatures, c->bLocalFeatures, triangulated);

    VisualDebugHook debugger;

    float ax = 0;
    float az = 0;
    float ay = 0;

    const  float scale = 1;
    
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
                 triangulated.at<float>(3, i) - az) * scale,
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[2],
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[1],
                a->image.data.at<Vec3b>(c->aLocalFeatures[i])[0]);
    }

    debugger.Draw();
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    if(n > 100) {
        n = 100;
    }

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);
        
    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    for(int i = 0; i < n; i++) {
        auto img = InputImageFromFile(files[i], false);

        img->originalExtrinsics = base * zero * img->originalExtrinsics.inv() * baseInv;
        cout << "Pushing " << img->id << endl;
        combiner.Push(img);
    }

    return 0;
}
