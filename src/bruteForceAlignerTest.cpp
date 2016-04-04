#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "recorder/iterativeBundleAligner.hpp"
#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "common/functional.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "recorder/alignmentGraph.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "stitcher/multiringStitcher.hpp"
#include "minimal/stereoConverter.hpp"
#include "minimal/imagePreperation.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    //cout << cv::getBuildInformation() << endl;
    bool outputUnaligned = true;
            
    SimpleSphereStitcher debugger(400);

    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv);

    RecorderGraph fullGraph = RecorderGraphGenerator::Generate(
            allImages[0]->intrinsics, 
            RecorderGraph::ModeTruncated, 
            1, 0, 4);


    BiMap<size_t, uint32_t> imagesToTargets, d;

    auto fullImages = fullGraph.SelectBestMatches(allImages, imagesToTargets);

    int n = fullImages.size();
    
    cout << "Selecting " << n << " images for further processing." << endl;

    auto miniImages = minimal::ImagePreperation::CreateMinifiedCopy(fullImages, 3);
    
    if(outputUnaligned) {
        auto res = debugger.Stitch(miniImages);
        imwrite("dbg/unaligned.jpg", res->image.data);
    }

    cout << "Performing in/extrinsics adjustment via center ring." << endl;
  
    vector<vector<InputImageP>> rings = fullGraph.SplitIntoRings(miniImages);

    size_t k = rings.size() / 2;

    minimal::ImagePreperation::SortById(rings[k]);
    RingCloser::CloseRing(rings[k]);

    if(outputUnaligned) {
        auto res = debugger.Stitch(miniImages);
        imwrite("dbg/center_ring_aligned.jpg", res->image.data);
    }

    cout << "Performing in/extrinsics adjustment bundle adjustment." << endl;

    IterativeBundleAligner aligner;
    aligner.Align(miniImages, fullGraph, imagesToTargets, 5, 0.5);

    minimal::ImagePreperation::CopyIntrinsics(miniImages, fullImages);
    minimal::ImagePreperation::CopyExtrinsics(miniImages, fullImages);

    auto res = debugger.Stitch(miniImages);
        imwrite("dbg/aligned.jpg", res->image.data);

    cout << "Create final stereo output." << endl;

    //Just for testing. 
    RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(fullGraph, 2);
    auto finalImages = halfGraph.SelectBestMatches(fullImages, imagesToTargets); 
    
    minimal::ImagePreperation::LoadAllImages(finalImages);
        
    minimal::StereoConverter::StitchAndWrite(
                finalImages, halfGraph, "aligned_stereo");

    return 0;
}
