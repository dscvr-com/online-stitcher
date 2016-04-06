#include <iostream>
#include <algorithm>
#include <opencv2/core/ocl.hpp>

#include "minimal/imagePreperation.hpp"
#include "common/intrinsics.hpp"
#include "recorder/recorder.hpp"
#include "io/io.hpp"
#include "stereo/monoStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    assert(n == 2);

    SelectionInfo a, b;

    a.image = images[0];
    b.image = images[1];

    RecorderGraphGenerator generator;
    RecorderGraph graph = generator.Generate(a.image->intrinsics, 
                        RecorderGraph::ModeAll);

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
