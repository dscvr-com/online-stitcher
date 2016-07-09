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
#include "stitcher/simpleSphereStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    const bool grayscale = false;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    Assert(n == 2);

    Mat imgA, imgB, corr;

    if(grayscale) {
        imgA = imread(files[0], CV_LOAD_IMAGE_GRAYSCALE);
        imgB = imread(files[1], CV_LOAD_IMAGE_GRAYSCALE);
    } else {
        imgA = imread(files[0]);
        imgB = imread(files[1]);
    }
    
    pyrDown(imgA, imgA);
    pyrDown(imgB, imgB);
    
    imwrite("dbg/inA.jpg", imgA);
    imwrite("dbg/inB.jpg", imgB);
    
    PlanarCorrelationResult res;
   
    if(grayscale) {
        res = BruteForcePlanarAligner<NormedCorrelator<AbsoluteDifference<uchar>>>::Align(imgA, imgB, corr, 0.5, 0.5);
    } else {
        res = BruteForcePlanarAligner<NormedCorrelator<AbsoluteDifference<cv::Vec3b>>>::Align(imgA, imgB, corr, 0.5, 0.5);
    }
    imwrite("dbg/corr.jpg", corr);

    cout << "Result: " << res.offset << ", variance: " << res.variance << endl;
    
    return 0;
}
