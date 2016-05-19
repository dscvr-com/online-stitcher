#include <iostream>
#include <opencv2/core/ocl.hpp>

#include "minimal/imagePreperation.hpp"
#include "recorder/visualStabilizer.hpp"

using namespace std;
using namespace optonaut;
using namespace cv;

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    Mat sensorInt = images[0]->originalExtrinsics; 

    VisualStabilizer stabilizer;

    for(size_t i = 0; i < n; i++) {
        stabilizer.Push(images[i]);
    }

    return 0;
}
