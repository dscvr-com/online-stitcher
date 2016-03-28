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
    bool outputUnalignedStereo = false;

    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv);

    RecorderGraph fullGraph = RecorderGraphGenerator::Generate(
            allImages[0]->intrinsics, 
            RecorderGraph::ModeTruncated, 
            1, 0, 4);

    RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(fullGraph, 2);

    BiMap<size_t, uint32_t> imagesToTargets;

    auto fullImages = fullGraph.SelectBestMatches(allImages, imagesToTargets);

    int n = fullImages.size();
    
    cout << "Selecting " << n << " images for further processing." << endl;

    auto miniImages = minimal::ImagePreperation::CreateMinifiedCopy(fullImages, 3);

    if(outputUnalignedStereo) {
        minimal::StereoConverter::StitchAndWrite(
                fullImages, fullGraph, "unaligned");
    }

    IterativeBundleAligner::Align(miniImages, fullGraph, imagesToTargets, 10, 0.5);

    minimal::ImagePreperation::CopyExtrinsics(miniImages, fullImages);

    //Just for testing. 
    auto finalImages = halfGraph.SelectBestMatches(fullImages, imagesToTargets); 
    
    minimal::ImagePreperation::LoadAllImages(finalImages);
        
    minimal::StereoConverter::StitchAndWrite(
                finalImages, halfGraph, "aligned");
    
    return 0;
}
