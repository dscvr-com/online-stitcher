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

    assert(n > 1);
        
    RecorderGraphGenerator generator;
    RecorderGraph graph = generator.Generate(images[0]->intrinsics, 
                            RecorderGraph::ModeAll, RecorderGraph::DensityDouble);

    for(size_t i = 0; i < n - 1; i++) {
        SelectionInfo a;
        SelectionInfo b;

        a.image = images[i];
        b.image = images[i + 1];

        graph.FindClosestPoint(a.image->originalExtrinsics, a.closestPoint);
        graph.FindClosestPoint(b.image->originalExtrinsics, b.closestPoint);

        if(a.closestPoint.globalId == b.closestPoint.globalId) {
            cout << "Skipping " << i << " and " << i + 1 << 
                " - same selection point." << endl;
            continue;
        }

        StereoImage res;
        MonoStitcher mono;

        mono.CreateStereo(a, b, res);
            
        imwrite("dbg/" + ToString(a.image->id) + "_stereo_a.jpg", res.A->image.data);
        imwrite("dbg/" + ToString(a.image->id) + "_stereo_b.jpg", res.B->image.data);
    }

    return 0;
}
