#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "imgproc/planarCorrelator.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);

    assert(n % 2 == 0);

    for(int i = 0; i < n; i += 2) {
        auto imgA = InputImageFromFile(files[i], false);
        auto imgB = InputImageFromFile(files[i + 1], false);

        auto base = Recorder::iosBase;
        auto zero = Recorder::iosZero;
        auto baseInv = base.t();

        imgA->originalExtrinsics = base * zero * imgA->originalExtrinsics.inv() * baseInv;
        imgB->originalExtrinsics = base * zero * imgB->originalExtrinsics.inv() * baseInv;
        imgA->adjustedExtrinsics = imgA->originalExtrinsics;
        imgB->adjustedExtrinsics = imgB->originalExtrinsics;
        
        SimpleSphereStitcher stitcher;
        auto scene = stitcher.Stitch({imgA, imgB});
        imwrite("dbg/" + ToString(i) + "_scene.jpg", scene->image.data);
        
        STimer timer;

        PairwiseCorrelator corr;
        auto result = corr.Match(imgA, imgB, 4, 4, false, 0.5, 1.8); 

        if(!result.valid) {
            cout << "Correlation: Rejected " << result.rejectionReason << "." << endl;
        }
      
        cout << "[" << i << "] BiasY: " << result.angularOffset.y << endl;
        cout << "[" << i << "] BiasX: " << result.angularOffset.x << endl;

        Mat rotY, rotX;
        CreateRotationY(result.angularOffset.y, rotY);
        CreateRotationX(-result.angularOffset.x, rotX);
        imgA->adjustedExtrinsics = imgA->adjustedExtrinsics * rotY * rotX; 
        imgA->originalExtrinsics = imgA->adjustedExtrinsics;
        
        scene = stitcher.Stitch({imgA, imgB});
        imwrite("dbg/" + ToString(i) + "_scene_aligned.jpg", scene->image.data);
    }
    return 0;
}
