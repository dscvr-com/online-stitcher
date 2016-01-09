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

    cout << "t: " << t.t() << endl;
    cout << "R: " << R << endl;

    float distortion[] = {0.0439, -0.0119, 0, 0};

    Mat Rect1, Rect2, P1, P2, Q, Distortion = Mat(4, 1, CV_32F, distortion);

    stereoRectify(scaledKAsFloat, Distortion, scaledKAsFloat, Distortion,
           a->image.size(), 
           R, t, Rect1, Rect2, P1, P2, Q);

    Mat triangulated;

    triangulatePoints(P1, P2, c->aLocalFeatures, c->bLocalFeatures, triangulated);

    VisualDebugHook debugger;

    float ax = 0;
    float az = 0;
    float ay = 0;

    const  float scale = 0.5;
    
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
        //pyrDown(img->image.data, img->image.data);
        //pyrDown(img->image.data, img->image.data);
        img->image = Image(img->image.data);

        img->originalExtrinsics = base * zero * img->originalExtrinsics.inv() * baseInv;
        img->adjustedExtrinsics = img->originalExtrinsics;

        cout << "Pushing " << img->id << endl;
        combiner.Push(img);
        if(i % 2 == 1)
            combiner.Flush();
    }

    return 0;
}
