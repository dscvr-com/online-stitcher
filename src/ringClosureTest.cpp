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
#include "recorder/streamingRecorderController.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "minimal/imagePreperation.hpp"
#include "recorder/ringCloser.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {

    //Setup reorder graph for center ring images.
    //Read images, sort center ring.
    //Get first and last images.
    //Approximate error, then distribute error equally where LAST image stays fixed (then we can assume the error to be 0). 

    cv::ocl::setUseOpenCL(false);

    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv);

    RecorderGraphGenerator generator;
    RecorderGraph graph = generator.Generate(allImages[0]->intrinsics, 
                        RecorderGraph::ModeCenter);

    vector<InputImageP> ring;

    ImageSelector selector(graph, [&ring] (const SelectionInfo &x) {
        ring.push_back(x.image);
    }, Vec3d(M_PI / 8, M_PI / 8, M_PI / 8));

    for(auto img : allImages) {
        selector.Push(img);
    }

    minimal::ImagePreperation::LoadAllImages(ring);

    SimpleSphereStitcher stitcher;
    auto scene = stitcher.Stitch(ring);
    imwrite("dbg/extracted_ring.jpg", scene->image.data);

    //Wooop woop ring closure
    //#######################
  
    RingCloser::CloseRing(ring); 

    //#######################

    scene = stitcher.Stitch(ring);
    imwrite("dbg/adjusted_ring.jpg", scene->image.data);

    return 0;
}
