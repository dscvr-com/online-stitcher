#include <iostream>
#include <algorithm>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "recorder/recorder.hpp"
#include "io/io.hpp"
#include "stereo/monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    assert(n == 2);

    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();
    
    SelectionInfo a, b;

    a.image = InputImageFromFile(files[0], false);
    b.image = InputImageFromFile(files[1], false);
    
    RecorderGraphGenerator generator;
    RecorderGraph graph = generator.Generate(a.image->intrinsics, 
                        RecorderGraph::ModeAll);

    a.image->originalExtrinsics = base * zero * 
        a.image->originalExtrinsics.inv() * baseInv;
    a.image->adjustedExtrinsics = a.image->originalExtrinsics;
    b.image->originalExtrinsics = base * zero * 
        b.image->originalExtrinsics.inv() * baseInv;
    b.image->adjustedExtrinsics = b.image->originalExtrinsics;

    graph.FindClosestPoint(a.image->originalExtrinsics, a.closestPoint);
    graph.FindClosestPoint(b.image->originalExtrinsics, b.closestPoint);

    AssertM(a.closestPoint.globalId != b.closestPoint.globalId, "Images have different selection points.");

    StereoImage res;
    MonoStitcher mono;

    mono.CreateStereo(a, b, res);
        
    imwrite("dbg/stereo_a.jpg", res.A->image.data);
    imwrite("dbg/stereo_b.jpg", res.B->image.data);

    return 0;
}
