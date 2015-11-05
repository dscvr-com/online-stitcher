#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "imgproc/planarCorrelator.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    assert(n == 2);

    Mat corr;

    auto imgA = InputImageFromFile(files[0], false);
    auto imgB = InputImageFromFile(files[1], false);

    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();
            
    imgA->originalExtrinsics = base * zero * imgA->originalExtrinsics.inv() * baseInv;
    imgB->originalExtrinsics = base * zero * imgB->originalExtrinsics.inv() * baseInv;

    Mat wa, wb;

    GetOverlappingRegion(imgA, imgB, imgA->image, imgB->image, wa, wb, 0);
    
    imwrite("dbg/overlapA.jpg", wa);
    imwrite("dbg/overlapB.jpg", wb);

    cvtColor(wa, wa, CV_RGB2GRAY);
    cvtColor(wb, wb, CV_RGB2GRAY);

    Point res = BruteForcePlanarAligner<NormedCorrelator<GemanMcClure<uchar, 128>>>::Align(wa, wb, corr, 0.5, 0.5);
    
    imwrite("dbg/corr.jpg", corr);

    cout << "Result: " << res << endl;
    
    return 0;
}
