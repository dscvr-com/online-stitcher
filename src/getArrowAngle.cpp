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

    int imageId_a = 0;
    int imageId_b = 1;

    // for 0 and 2954 dist : 1.39646 , rotation : 6.28319
    
            
    SimpleSphereStitcher debugger;
    cout << "load image and extrinsics " ;
    auto allImages = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv);
    Mat rvec;
    Mat image_a = allImages[imageId_a]->adjustedExtrinsics;
    Mat image_b = allImages[imageId_b]->adjustedExtrinsics;

    Mat last_image = ConvertFromStitcher(image_a);



    ExtractRotationVector(image_a.inv() * image_b, rvec);

    double angularDist = rvec.at<double>(1);
    cout << "angularDist : " << ToString(angularDist) << endl;

    // get angle of rotation
    double angleOfRotation = GetAngleOfRotation ( image_a, image_b);
    cout << "angularOfRotation : " << ToString(angleOfRotation) << endl;


    double dist_X = GetDistanceX(image_a, image_b);
    cout << "dist_X : " << ToString(dist_X) << endl;

    double dist_Y = GetDistanceY(image_a, image_b);
    cout << "dist_Y : " << ToString(dist_Y) << endl;
    
    double dist_Z = GetDistanceZ(image_a, image_b);
    cout << "dist_Z : " << ToString(dist_Z) << endl;

    float angle = atan2(dist_X, dist_Y);
    cout << "angle : " << ToString(angle) << endl;









    return 0;
}
