#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/video/tracking.hpp>
//#include "../common/NEON_2_SSE.h" // TODO: disable on ARM
//#include <arm_neon.h>

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

    void FlowBlender::CalculateFlow(
            const Mat &a, const Mat &b, 
            const Point &aTl, const Point &bTl,
            Mat &flow, 
            Point &offset, const bool reCalcOffset) const {

        STimer t;

        Rect aRoi(aTl, a.size());
        Rect bRoi(bTl, b.size());

        Rect overlap = aRoi & bRoi;

        flow = Mat(b.size(), CV_32FC2, Scalar::all(0.f));

        if(overlap.width == 0 || overlap.height == 0)
            return;
      
        // Corner case: Super-big image for border images.  
        const bool tooBig = a.cols > destRoi.width / 4 || b.cols > destRoi.width / 4;

        if(useFlow && !tooBig) {
            Rect aOverlap(overlap.tl() - aTl, overlap.size());
            Rect bOverlap(overlap.tl() - bTl, overlap.size());

            Mat aOverlapImg = a(aOverlap);
            Mat bOverlapImg = b(bOverlap);
            
            Mat _flow = flow(bOverlap);

            if(reCalcOffset) { 
                typedef PyramidPlanarAligner<
                    NormedCorrelator<LeastSquares<Vec3b>>
                > AlignerToUse;

                Mat corr; //Debug image used to print the correlation result.  
                PlanarCorrelationResult result = AlignerToUse::Align(
                        aOverlapImg, bOverlapImg, corr, 0.2, 0.01, 0);

                offset = result.offset;

                Log << "New offset: " << offset;
            }

            t.Tick("New Offset Calculation");

            Rect roiA(offset.x / -2, offset.y / -2, 
                    aOverlapImg.cols, aOverlapImg.rows);

            Rect roiB(offset.x / 2, offset.y / 2, 
                    bOverlapImg.cols, bOverlapImg.rows);

            Rect overlappingArea = roiA & roiB;

            Rect overlapAreaA(overlappingArea.tl() + roiA.tl(),
                    overlappingArea.size()); 

            Rect overlapAreaB(overlappingArea.tl() + roiB.tl(),
                    overlappingArea.size());        

            aOverlapImg = aOverlapImg(overlapAreaA);
            bOverlapImg = bOverlapImg(overlapAreaB);

            _flow = _flow(overlapAreaB);

            UMat ig, dg;

            cvtColor(aOverlapImg, dg, COLOR_BGR2GRAY);
            cvtColor(bOverlapImg, ig, COLOR_BGR2GRAY);

            //pyrDown(dg, dg);
            //pyrDown(ig, ig);
           
            UMat tmp(dg.size(), CV_32FC2);    

            calcOpticalFlowFarneback(dg, ig, tmp, 
                    0.5, // Pyr Scale
                    1, // Levels
                    5, // Winsize
                    4, // Iterations
                    5, // Poly N 
                    1.1, // Poly Sigma
                    0); // Flags

            //pyrUp(tmp, tmp);
            tmp.copyTo(_flow);
                
            if(debug) {
                static int dbgCtr2 = 0;
                imwrite("dbg/" + ToString(dbgCtr2) + "_b.jpg", b);
                imwrite("dbg/" + ToString(dbgCtr2) + "_a.jpg", a);
                imwrite("dbg/" + ToString(dbgCtr2) + "_ob.jpg", bOverlapImg);
                imwrite("dbg/" + ToString(dbgCtr2) + "_oa.jpg", aOverlapImg);
                dbgCtr2++;
            }
        }
        
        cv::add(flow, Scalar(offset.x, offset.y), flow);
/*
        for (int y = 0; y < b.rows; ++y)
        {
            for (int x = 0; x < b.cols; ++x)
            {
                Vec2f d = flow.at<Vec2f>(y, x);
                flow.at<Vec2f>(y, x) = Vec2f(d(0) + offset.x, d(1) + offset.y);
            }
        }
  */      
        t.Tick("Flow Calculation");
    }


    float neon_clamp ( float val, float minval, float maxval )
    {
        // Branchless NEON clamp.
        // return vminq_f32(vmaxq_f32(val, minval), maxval);
        return std::min(std::max(val, minval), maxval);
    }

    void FlowBlender::Feed(const Mat &img, const Mat &flow, const Point &tl)
    {
        AssertEQ(img.type(), CV_8UC3);
        AssertEQ(flow.type(), CV_32FC2);

        STimer t;
        static int dbgCtr = 0;

        Mat wmDest(img.size(), CV_32F, Scalar::all(0));

        const Rect sourceRoi(tl - destRoi.tl(), img.size());
        
        cv::detail::createWeightMap(destMask(sourceRoi), sharpness, wmDest);
        
        int dx = sourceRoi.x;
        int dy = sourceRoi.y;
        
        t.Tick("Blending Preperation");

        if(debug) {
            Mat flowViz;
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
            imwrite("dbg/" + ToString(dbgCtr) + "_flow.jpg", flowViz);

        }

        if(debug) {

            imwrite("dbg/" + ToString(dbgCtr) + "_in.jpg", img);
            imwrite("dbg/" + ToString(dbgCtr) + "_dest.jpg", dest(sourceRoi));
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_mask.jpg", destMask(sourceRoi));
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_weight_map.jpg", (Mat)(wmDest * 255));
            
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_full.jpg", dest);
            imwrite("dbg/" + ToString(dbgCtr) + "_dest_mask_full.jpg", destMask);
        } 

        Mat temp(img.size(), CV_8UC3);

        int w = img.cols;
        int h = img.rows;
        int dw = dest.cols;
        int dh = dest.rows;

        Mat imgMapX(w, h, CV_32F);
        Mat imgMapY(w, h, CV_32F);
        Mat zero(w, h, CV_32F, Scalar(0.f));
        
        Mat destMapX(w, h, CV_32F);
        Mat destMapY(w, h, CV_32F);

        // The following code block converts
        // the flow image to a map for remapping 
        // using raw pointers. 
        float* pImgMapX = (float*)imgMapX.ptr();
        float* pImgMapY = (float*)imgMapY.ptr();
        float* pDestMapX = (float*)destMapX.ptr();
        float* pDestMapY = (float*)destMapY.ptr();
        float* pFlow = (float*)flow.ptr();
        float* pwmDest = (float*)wmDest.ptr();
        
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
               
                float wcd = *pwmDest++;
                float wcs = 1.f - wcd;

                // Convert flow to remap representation.
                // Also add weights in one go. 
                // Translation of source position is proportional
                // to dest weight and vice-versa. 
                float imgDx = x + *pFlow * wcd;
                float destDx = dx + x - *pFlow++ * wcs;
                float imgDy = y + *pFlow * wcd;
                float destDy = dy + y - *pFlow++ * wcs;

                // Check mapping - if out-of-bounds we use Identity mapping
                // Todo: Might want to check mask
                imgDx = imgDx < 0 || imgDx >= w ? x : imgDx;
                imgDy = imgDy < 0 || imgDy >= h ? y : imgDy;

                destDx = destDx < 0 || destDx >= dw ? x : destDx;
                destDy = destDy < 0 || destDy >= dh ? y : destDy;

                *pImgMapX++ = imgDx;
                *pImgMapY++ = imgDy;
                *pDestMapX++ = destDx;
                *pDestMapY++ = destDy;
            }
        }

        // Remap the images as calculated above.
        // That is 4 times faster than direct sampling. 
        Mat remappedImg(w, h, CV_8UC3);
        Mat remappedDest(w, h, CV_8UC3);
        
        cv::remap(img, remappedImg, imgMapX, imgMapY, INTER_LINEAR);
        cv::remap(dest, remappedDest, destMapX, destMapY, INTER_LINEAR);

        /// Now blend the remapped images. 
        pwmDest = (float*)wmDest.ptr();
        uchar* pRemappedImg = remappedImg.ptr();
        uchar* pRemappedDest = remappedDest.ptr();
        uchar* pTemp = temp.ptr();
        
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                float wcd = *pwmDest++;
                float wcs = 1.f - wcd;

                for(int c = 0; c < 3; c++) {
                    *pTemp++ = (uchar)(wcd * (*pRemappedDest++) + wcs * (*pRemappedImg++));
                }
            }
        }

        if(debug) {
            imwrite("dbg/" + ToString(dbgCtr) + "_blended.jpg", temp);
            dbgCtr++;
        }
        
        t.Tick("Blending Loop");

        // Commit the results to the destination image. 
        temp.copyTo(dest(sourceRoi));
        destMask(sourceRoi).setTo(Scalar(255));

        existingCores.push_back(sourceRoi);
        t.Tick("Blending Commit");
    }

    Point FlowBlender::dummyFlow = Point(0, 0);
}

