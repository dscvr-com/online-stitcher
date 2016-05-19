#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/sfm.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "imgproc/pairwiseVisualAligner.hpp"
#include "minimal/imagePreperation.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner;

void MatchImages(const InputImageP &a, const InputImageP &b) {
    aligner.FindCorrespondence(a, b); 
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    combiner.Process(minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false));

    int i = 0;
    for(auto chain : aligner.GetFeatureChains()) {
        i++;
        if(chain.size() > 2) {
            cout << "Chain " << i << ": ";

            for(auto ref : chain) {
                cout << ref.imageId << " -> ";
            }
            cout << endl;
        }
    }

    return 0;
}
