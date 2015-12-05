#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>
#include <srba.h>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "imgproc/pairwiseVisualAligner.hpp"

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

    int n = argc - 1;
    vector<string> files;

    if(n > 100) {
        n = 100;
    }

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);
        
    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    for(int i = 0; i < n - 1; i++ ) {
        auto img = InputImageFromFile(files[i], false);
        pyrDown(img->image.data, img->image.data);
        img->originalExtrinsics = base * zero * img->originalExtrinsics.inv() * baseInv;
        combiner.Push(img);
    }

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
