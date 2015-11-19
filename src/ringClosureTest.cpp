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

    //Setup reorder graph for center ring images.
    //Read images, sort center ring.
    //Get first and last images.
    //Approximate error, then distribute error equally where LAST image stays fixed (then we can assume the error to be 0). 

    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);

    cout << "Generating graph." << endl;

    auto image0 = InputImageFromFile(files[0], false);

    RecorderGraphGenerator generator;
    RecorderGraph graph = generator.Generate(image0->intrinsics, 
                        RecorderGraph::ModeCenter);
    cout << "Graph generated" << endl;
    RecorderController controller(graph);

    SelectionInfo currentBest;
    SimpleSphereStitcher stitcher;
        
    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();

    size_t imagesToRecord = graph.Size();
    vector<InputImageP> ring;
    ring.reserve(imagesToRecord);

    for(int i = 0; i < n; i++) {
        auto image = InputImageFromFile(files[i], false);
        image->originalExtrinsics = base * zero * image->originalExtrinsics.inv() * baseInv;
        image->originalExtrinsics = image->originalExtrinsics.clone();
        image->adjustedExtrinsics = image->originalExtrinsics.clone();
        AssertEQ(image->adjustedExtrinsics.cols, 4);
        AssertEQ(image->adjustedExtrinsics.rows, 4);
        AssertEQ(image->adjustedExtrinsics.type(), CV_64F);

        AssertGT(image->image.data.cols, 0);
        AssertGT(image->image.data.rows, 0);

        if(!controller.IsInitialized())
            controller.Initialize(image->adjustedExtrinsics);
        
        SelectionInfo current = controller.Push(image, false);

        if(current.isValid) {
            if(currentBest.isValid &&
                    current.closestPoint.globalId != 
                    currentBest.closestPoint.globalId) {
                ring.push_back(currentBest.image);
        
                AssertEQ(currentBest.image->adjustedExtrinsics.cols, 4);
        AssertEQ(currentBest.image->adjustedExtrinsics.rows, 4);
        AssertEQ(currentBest.image->adjustedExtrinsics.type(), CV_64F);

        AssertGT(currentBest.image->image.data.cols, 0);
        AssertGT(currentBest.image->image.data.rows, 0);

                cout << "Found " << ring.size() << " images." << endl;
            }
            
            currentBest = current;

            if(ring.size() >= imagesToRecord) 
                i = n;
        }
    }

    auto scene = stitcher.Stitch(ring);
    imwrite("dbg/extracted_ring.jpg", scene->image.data);

    return 0;
}
