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
    
    cv::ocl::setUseOpenCL(false);

    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);

    SimpleSphereStitcher stitcher(800);
    auto scene = stitcher.Stitch(allImages, false, true);

    imwrite("dbg/result.jpg", scene->image.data);

    return 0;
}
