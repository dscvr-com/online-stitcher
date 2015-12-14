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
    StreamingRecorderController controller(graph);

    SelectionInfo currentBest;
        
    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();

    size_t imagesToRecord = graph.Size();
    vector<InputImageP> ring;
    ring.reserve(imagesToRecord);

    for(int i = 0; i < n; i++) {
        auto image = InputImageFromFile(files[i], false);
        image->originalExtrinsics = base * zero * image->originalExtrinsics.inv() * baseInv;
        image->adjustedExtrinsics = image->originalExtrinsics;

        if(!controller.IsInitialized())
            controller.Initialize(image->adjustedExtrinsics);
        
        SelectionInfo current = controller.Push(image, false);

        if(current.isValid) {
            if(currentBest.isValid &&
                    current.closestPoint.globalId != 
                    currentBest.closestPoint.globalId) {
                ring.push_back(currentBest.image);
        
                cout << "Found " << ring.size() << " images." << endl;
            }
            
            currentBest = current;

            if(ring.size() >= imagesToRecord) 
                i = n;
        }
    }
    
    SimpleSphereStitcher stitcher;
    auto scene = stitcher.Stitch(ring);
    imwrite("dbg/extracted_ring.jpg", scene->image.data);

    //Wooop woop ring closure
    //#######################
   
    PairwiseCorrelator corr;

    auto result = corr.Match(ring.front(), ring.back()); 

    n = ring.size();

    for(int i = 0; i < n; i++) {
        double ydiff = result.angularOffset.x * (1.0 - ((double)i) / ((double)n));
        Mat correction;
        CreateRotationY(ydiff, correction);
        ring[i]->adjustedExtrinsics = correction * ring[i]->adjustedExtrinsics;
    }

    //#######################

    scene = stitcher.Stitch(ring);
    imwrite("dbg/adjusted_ring.jpg", scene->image.data);

    return 0;
}
