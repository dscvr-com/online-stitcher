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
#include "stitcher/simplePlaneStitcher.hpp"

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

    std::sort(files.begin(), files.end(), CompareByFilename);

    assert(n % 2 == 0);

    for(int i = 0; i < n; i += 2) {
        Mat corr;

        auto imgA = InputImageFromFile(files[i], false);
        auto imgB = InputImageFromFile(files[i + 1], false);

        auto base = Recorder::iosBase;
        auto zero = Recorder::iosZero;
        auto baseInv = base.t();

        imgA->originalExtrinsics = base * zero * imgA->originalExtrinsics.inv() * baseInv;
        imgB->originalExtrinsics = base * zero * imgB->originalExtrinsics.inv() * baseInv;
        imgA->adjustedExtrinsics = imgA->originalExtrinsics;
        imgB->adjustedExtrinsics = imgB->originalExtrinsics;
        
        SimpleSphereStitcher stitcher;
        auto scene = stitcher.Stitch({imgA, imgB});
        imwrite("dbg/" + ToString(i) + "_scene.jpg", scene->image.data);

        Mat wa, wb;

        GetOverlappingRegion(imgA, imgB, imgA->image, imgB->image, wa, wb, 0);

        pyrDown(wa, wa);
        pyrDown(wb, wb);
        pyrDown(wa, wa);
        pyrDown(wb, wb);
        
        imwrite("dbg/" + ToString(i) + "_overlapA.jpg", wa);
        imwrite("dbg/" + ToString(i) + "_overlapB.jpg", wb);

        //cvtColor(wa, wa, CV_RGB2GRAY);
        //cvtColor(wb, wb, CV_RGB2GRAY);

        Point res = BruteForcePlanarAligner<NormedCorrelator<AbsoluteDifference<Vec3b>>>::Align(wa, wb, corr, 0.25, 0.25);
        
        cout << "Result: " << res << endl;

        SimplePlaneStitcher planeStitcher;
        scene = planeStitcher.Stitch({make_shared<Image>(wa), make_shared<Image>(wb)}, 
                {Point(-res.x, res.y), Point(0, 0)});
        imwrite("dbg/" + ToString(i) + "_overlap_aligned.jpg", scene->image.data);

        //TODO 
        //imgB->adjustedExtrinsics.at<double>(0, 3) += res.x;
        //imgB->adjustedExtrinsics.at<double>(1, 3) += res.y;
        
        scene = stitcher.Stitch({imgA, imgB});
        imwrite("dbg/" + ToString(i) + "_scene_aligned.jpg", scene->image.data);

        for(int i = 0; i < corr.cols; i++) {
            for(int j = 0; j < corr.rows; j++) {
                if(corr.at<float>(j, i) > 255) {
                    cout << "Warning, correlation matrix contains values >255. Image output might not be useful.";
                    i = corr.cols;
                    j = corr.rows;
                }
            }
        }
        
        imwrite("dbg/corr.jpg", corr);
    }
    return 0;
}
