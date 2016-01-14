#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>
#define CERES_FOUND
#include <opencv2/sfm/reconstruct.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "minimal/imagePreperation.hpp"
#include "debug/visualDebugHook.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;
using namespace cv::sfm;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner;

void MatchImages(const InputImageP &a, const InputImageP &b) {
    aligner.FindCorrespondence(a, b); 
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(
            argc, argv, true, 20, 1);
    images = minimal::ImagePreperation::CreateMinifiedCopy(images, 1);
    int n = images.size();

    std::map<size_t, int> imagesToLocalId;
    auto byId = minimal::ImagePreperation::CreateImageMap(images);
    std::map<int, InputImageP> byGlobalId;;

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    for(int i = 0; i < n; i++) {
        auto img = images[i]; 

        imagesToLocalId[img->id] = i;
        byGlobalId[i] = img;

        cout << "Pushing " << img->id << endl;
        combiner.Push(img);
    }
    combiner.Flush();

    auto chains = aligner.GetFeatureChains(); 
    int k = chains.size(); 

    //Outer array -> frames, inner array, points within frame
    vector<Mat> trackingFeatures(n); 

    for(int i = 0; i < n; i++) {
        trackingFeatures[i] = Mat(2, k, CV_64F, Scalar::all(-1));
    }

    for(int i = 0; i < k; i++) {
        int q = chains[i].size();
        for(int j = 0; j < q; j++) {
            int globalImageId = chains[i][j].imageId;
            int localFeatureId = chains[i][j].featureIndex;
            int globalFeatureId = i;

            auto features = aligner.GetFeaturesById(globalImageId);
            auto pt = features.keypoints[localFeatureId].pt; 

            trackingFeatures[globalImageId].at<double>(0, globalFeatureId) = pt.x; 
            trackingFeatures[globalImageId].at<double>(1, globalFeatureId) = pt.y; 
        }
    }

    bool isProjective = true;
    vector<Mat> rEst, tEst, triangulated;
    Mat K;
    ScaleIntrinsicsToImage(images[0]->intrinsics, images[0]->image.size(), K);
    reconstruct(trackingFeatures, rEst, tEst, K, triangulated, isProjective); 

    VisualDebugHook debugger;
    
    for(size_t i = 0; i < triangulated.size(); i++) {
        cout << triangulated[i].at<double>(0) << ", " <<
            triangulated[i].at<double>(1) << ", " <<
            triangulated[i].at<double>(2) << endl;

        //InputImageP a = byGlobalId.at(i);

        debugger.PlaceFeature(
                triangulated[i].at<double>(0), 
                triangulated[i].at<double>(1), 
                triangulated[i].at<double>(2),
                0xFF, 0x00, 0x00); //TODO: Get Color of pixel in first image that contains the feature
    }

    debugger.Draw();

    return 0;
}
