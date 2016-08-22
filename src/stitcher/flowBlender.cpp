#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

#include "flowBlender.hpp"
#include "../common/assert.hpp"
#include "../common/support.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"

static const bool debug = false;

namespace optonaut {

    void FlowBlender::Prepare(const Rect &roi) {
        destRoi = roi;
        dest.create(roi.size(), CV_8UC3);
        dest.setTo(Scalar(0));;
        destMask.create(roi.size(), CV_8UC1);
        destMask.setTo(Scalar(0));;
   }

    void FlowBlender::Feed(const Mat &img, const Mat &flow_, const Point &tl)
    {
        static int dbgCtr = 0;
        typedef PyramidPlanarAligner<NormedCorrelator<LeastSquares<Vec3b>>> AlignerToUse;
        AssertEQ(img.type(), CV_8UC3);

        Mat wmDest(img.size(), CV_32F, Scalar::all(0));
        Mat wmSource(img.size(), CV_32F, Scalar::all(0));

        const Rect sourceRoi(tl - destRoi.tl(), img.size());

        cv::detail::createWeightMap(destMask(sourceRoi), sharpness, wmDest);

        Mat flow(img.size(), CV_32FC2, Scalar::all(0));

        bool useOwnFlow = true;

        if(!useOwnFlow) {
            AssertEQ(flow_.type(), CV_32FC2);
            flow_.copyTo(flow);
        } else {

            Rect core(0, 0, 0, 0);

            for(const Rect &dstCore: existingCores) 
            {
                Rect cand = dstCore & sourceRoi; 
              
                if(cand.width * cand.height > core.width * core.height)
                   core = cand; 
            }

            
            cout << core << endl;

            if(core.width != 0 && core.height != 0) {
                
                Rect imgRegion(core.tl() - tl + destRoi.tl(), core.size());

                Mat corr; //Debug image used to print the correlation result.  
                PlanarCorrelationResult result = AlignerToUse::Align(dest(core), img(imgRegion), corr, 0.25, 0.01, 0);

                Mat dg;
                Mat imgGray;

                Rect roiA(result.offset.x / -2, result.offset.y / -2, core.width, core.height);
                Rect roiB(result.offset.x / 2, result.offset.y / 2, core.width, core.height);

                Rect overlappingArea = roiA & roiB;

                Rect overlapAreaA(overlappingArea.tl() + roiA.tl(), overlappingArea.size()); 
                Rect overlapAreaB(overlappingArea.tl() + roiB.tl(), overlappingArea.size());        


                cvtColor(img(imgRegion)(overlapAreaB), imgGray, COLOR_BGR2GRAY);
                cvtColor(dest(core)(overlapAreaA), dg, COLOR_BGR2GRAY);
                //calcOpticalFlowFarneback(grayA, grayB, cutFlow, 0.5, 3, 4, 3, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
                calcOpticalFlowFarneback(imgGray, dg, flow(imgRegion)(overlapAreaB), 0.5, 3, 4, 3, 7, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
                
                for (int y = 0; y < flow.rows; ++y)
                {
                    for (int x = 0; x < flow.cols; ++x)
                    {
                        auto d = flow.at<Vec2f>(y, x);
                        flow.at<Vec2f>(y, x) = Vec2f(d(0) + result.offset.x, d(1) + result.offset.y);
                   }
                }

                if(debug) {
                    imwrite("dbg/" + ToString(dbgCtr) + "_overlapB.jpg", dg);
                    imwrite("dbg/" + ToString(dbgCtr) + "_overlapA.jpg", imgGray);
                }
            }
        }
        int dx = tl.x - destRoi.x;
        int dy = tl.y - destRoi.y;

        Mat flowViz;

        if(debug) {
            flowViz = Mat(flow.size(), CV_8UC3, Scalar::all(0));

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

        }
        
        for (int y = 0; y < img.rows; ++y)
        {
            for (int x = 0; x < img.cols; ++x)
            {
                Vec2f d = flow.at<Vec2f>(y, x);

                int sfx = x + d(0);
                int sfy = y + d(1);
                int dfx = x + dx - d(0);
                int dfy = y + dy - d(1);

                if(sfx < 0 || sfy < 0 || dfx < 0 || dfy < 0 || 
                   sfx >= img.cols || sfy >= img.rows || dfx >= dest.cols || dfy >= dest.rows ||
                   destMask.at<uchar>(dfy, dfx) == 0) {

                    flow.at<Vec2f>(y, x) = Vec2f(0, 0);   
                    if(debug) {
                        flowViz.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
                    }
                }
            }
        }

        if(debug) {

            imwrite("dbg/" + ToString(dbgCtr) + "_flow.jpg", flowViz);
            imwrite("dbg/" + ToString(dbgCtr) + "_in.jpg", img);
            imwrite("dbg/" + ToString(dbgCtr) + "_dest.jpg", dest(sourceRoi));
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_mask.jpg", destMask(sourceRoi));
            imwrite("dbg/" + ToString(dbgCtr) + "_in_weight_map.jpg", (Mat)(wmSource * 255));
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_weight_map.jpg", (Mat)(wmDest * 255));
            
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_full.jpg", dest);
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_mask_full.jpg", destMask);
        } 

        Mat temp(img.size(), CV_8UC3);

        for (int y = 0; y < img.rows; ++y)
        {
            for (int x = 0; x < img.cols; ++x)
            {
                float wcd = wmDest.at<float>(y, x);
                //float ws = wmSource.at<float>(y, x); 
                //float norm = wd + ws;

                //if(norm == 0) { // no input. 
                //    dest.at<Vec3b>(y, x) = Vec3b(0, 255, 0);
                //    continue;
                //}

                //wd = wd / norm;
                //ws = ws / norm;
                float wcs = 1 - wcd;
              
                float wpd = wcd;
                float wps = 1 - wpd;

                Vec2f d = flow.at<Vec2f>(y, x);
               
                temp.at<Vec3b>(y, x) = 
                    Sample<Vec3b>(img, x + d(0) * wpd, y + d(1) * wpd) * wcs + 
                    Sample<Vec3b>(dest, dx + x - d(0) * wps, dy + y - d(1) * wps) * wcd;
                //temp.at<Vec3b>(y, x) = Vec3b(wd * 255, wd * 255, wd * 255);

                destMask.at<uchar>(dy + y, dx + x) = 255; 
            }
        }

        if(debug) {
            imwrite("dbg/" + ToString(dbgCtr) + "_blended.jpg", temp);
            dbgCtr++;
        }

        temp.copyTo(dest(Rect(dx, dy, temp.cols, temp.rows)));

        existingCores.push_back(sourceRoi);
    }
}

