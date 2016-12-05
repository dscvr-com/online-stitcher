#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>

#include "minimal/imagePreperation.hpp"
#include "common/static_timer.hpp"
#include "common/logger.hpp"
#include "math/support.hpp"
#include "imgproc/planarCorrelator.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> AlignerToUse;

void VisualizeFlow(const Mat &flow, const string& name) {
    Mat flowViz(flow.size(), CV_8UC3, Scalar::all(0));

    for(int k = 0; k < flow.cols; k++) {
        for(int j = 0; j < flow.rows; j++) {
            auto f = flow.at<Vec2f>(j, k);
            flowViz.at<Vec3b>(j, k) = 
                Vec3b(
                        std::min(255, (int)std::abs(f(0)) * 10), 
                        std::min(255, (int)std::abs(f(1)) * 10), 
                        0);
        }
    }

    imwrite(name, flowViz);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    Assert(n % 2 == 0);

    for(size_t i = 0; i < n - 1; i += 1) {
        auto imgA = images[i];
        auto imgB = images[i + 1];

        Mat overlapA, overlapB; 
        Point q;


        GetOverlappingRegion(imgA, imgB, imgA->image, imgB->image, 
                overlapA, overlapB, 0, q);
        
        Mat corr; //Debug image used to print the correlation result.  
        PlanarCorrelationResult result = AlignerToUse::Align(overlapA, overlapB, corr, 0.25, 0.01, 0);

        cout << "Corr: " << result.offset << endl;

        Rect roiA(result.offset.x / -2, result.offset.y / -2, overlapA.cols, overlapA.rows);
        Rect roiB(result.offset.x / 2, result.offset.y / 2, overlapB.cols, overlapB.rows);

        Rect overlappingArea = roiA & roiB;

        Rect overlapAreaA(overlappingArea.tl() + roiA.tl(), overlappingArea.size()); 
        Rect overlapAreaB(overlappingArea.tl() + roiB.tl(), overlappingArea.size());        

        Mat grayA, grayB;

        imwrite("dbg/a.jpg", overlapA(overlapAreaA));
        imwrite("dbg/b.jpg", overlapB(overlapAreaB));        

        cvtColor(overlapA(overlapAreaA), grayA, COLOR_BGR2GRAY);
        cvtColor(overlapB(overlapAreaB), grayB, COLOR_BGR2GRAY);

        Mat flow = Mat(grayA.size(), CV_32FC2);

        STimer t;

        t.Reset();
        calcOpticalFlowFarneback(grayA, grayB, flow, 
                0.5, // Pyr Scale
                5, // Levels
                5, // Winsize
                5, // Iterations
                7, // Poly N 
                1.5, // Poly Sigma
                0); // Flags
        t.Tick("Farneback flow");
        
        VisualizeFlow(flow, "dbg/farnebeck.jpg");
       /* 
        Ptr<DenseOpticalFlow> tvl1 = optflow::createOptFlow_DeepFlow();
        t.Reset();
        tvl1->calc(grayA, grayB, flow);
        t.Tick("Deep flow");
        
        VisualizeFlow(flow, "dbg/simple.jpg");
        */

    }
    return 0;
}
