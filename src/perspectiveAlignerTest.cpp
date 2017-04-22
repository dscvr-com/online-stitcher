#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "minimal/imagePreperation.hpp"
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

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    Assert(n % 2 == 0);

    for(size_t i = 0; i < n; i += 2) {
        auto imgA = images[i];
        auto imgB = images[i + 1];

        SimpleSphereStitcher stitcher;
        auto scene = stitcher.Stitch({imgA, imgB});
        imwrite("dbg/" + ToString(i) + "_scene_1_unaligned.jpg", scene->image.data);
        

        PairwiseCorrelator corr;
        STimer timer;
        auto result = corr.Match(imgA, imgB, 4, 4, false, 0.5, 1.8); 
        timer.Tick("Match");

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
        imwrite("dbg/" + ToString(i) + "_scene_2_aligned.jpg", scene->image.data);
    }
    return 0;
}
