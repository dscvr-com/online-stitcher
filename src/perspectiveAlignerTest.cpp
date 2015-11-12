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
        
        STimer timer;

        Mat wa, wb;

        Point appliedBorder;
        Rect overlappingRoi = GetOverlappingRegion(imgA, imgB, imgA->image, imgB->image, wa, wb, imgA->image.cols * 0.2, appliedBorder);

        timer.Tick("Overlap");

        Point res = PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>>::Align(wa, wb, 0.25, 0.25, 1);
        Point correctedRes = res + appliedBorder; 
        
        timer.Tick("Aligned");

        cout << correctedRes << ", " << overlappingRoi << endl;
       
       //TODO: Clean this sandbox up.  

        cout << "Result: " << res << endl;
        
        imwrite("dbg/" + ToString(i) + "_corr.jpg", corr);
        imwrite("dbg/" + ToString(i) + "_overlapA.jpg", wa);
        imwrite("dbg/" + ToString(i) + "_overlapB.jpg", wb);


        SimplePlaneStitcher planeStitcher;
        scene = planeStitcher.Stitch({make_shared<Image>(wa), make_shared<Image>(wb)},              {Point(res.x, res.y), Point(0, 0)});
        imwrite("dbg/" + ToString(i) + "_overlap_aligned.jpg", scene->image.data);
        
        double h = imgB->intrinsics.at<double>(0, 0) * (imgB->image.cols / (imgB->intrinsics.at<double>(0, 2) * 2));
        double olXA = (overlappingRoi.x + correctedRes.x - imgB->image.cols / 2) / h;
        double olXB = (overlappingRoi.x - imgB->image.cols / 2) / h;
        double corrAngleY = sin(olXA) - sin(olXB);
        
        cout << "BiasY: " << corrAngleY << endl;

        Mat rotY;
        CreateRotationY(corrAngleY, rotY);
        imgA->adjustedExtrinsics = rotY * imgA->adjustedExtrinsics; 
        
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
