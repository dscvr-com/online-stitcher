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
#include "stitcher/multiRingStitcher.hpp"
#include "minimal/stereoConverter.hpp"
#include "minimal/imagePreperation.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    bool outputUnalignedStereo = false;

    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv);

    RecorderGraph fullGraph = RecorderGraphGenerator::Generate(
            allImages[0]->intrinsics, 
            RecorderGraph::ModeTruncated, 
            1, 0, 4);

    BiMap<size_t, uint32_t> imagesToTargets;

    auto fullImages = fullGraph.SelectBestMatches(allImages, imagesToTargets);

    int n = fullImages.size();
    
    cout << "Selecting " << n << " images for further processing." << endl;

    auto miniImages = minimal::ImagePreperation::CreateMinifiedCopy(fullImages);

    if(outputUnalignedStereo) {
        minimal::StereoConverter::StitchAndWrite(
                fullImages, fullGraph, "unaligned");
    }

    IterativeBundleAligner::Align(miniImages, fullGraph, imagesToTargets);

    minimal::ImagePreperation::CopyExtrinsics(miniImages, fullImages);

    //Just for testing. 
    auto finalImages = fullGraph.SelectBestMatches(fullImages, imagesToTargets); 
    
    minimal::ImagePreperation::LoadAllImages(finalImages);
        
    minimal::StereoConverter::StitchAndWrite(
                fullImages, fullGraph, "aligned");
    
    return 0;
}
