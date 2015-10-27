#include "dynamicSeamer.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <vector>
#include "support.hpp"
#include "static_timer.hpp"

#define REMAPX(x,y)  (vertical ? x : y)
#define REMAPY(x,y)  (vertical ? y : x)

using namespace std;
using namespace cv;

namespace optonaut {

int DynamicSeamer::debugId = 0;

template <bool vertical>
void DynamicSeamer::Find(Mat& imgA, Mat &imgB, Mat &maskA, Mat &maskB, 
        const Point &tlAIn, const Point &tlBIn, int overlap, int id)
{
    STimer seamTimer;
    static const bool debug = false;

    Point tlA;
    Point tlB;
    int acols, bcols, arows, brows;

    if(vertical) {
        tlA = tlAIn;
        tlB = tlBIn;
        acols = imgA.cols;
        arows = imgA.rows;
        bcols = imgB.cols;
        brows = imgB.rows;
    } else  {
        tlA = Point(tlAIn.y, tlAIn.x);
        tlB = Point(tlBIn.y, tlBIn.x);
        acols = imgA.rows;
        arows = imgA.cols;
        bcols = imgB.rows;
        brows = imgB.cols;
    }
        
    Rect roi = Rect(tlA.x, tlA.y, acols, arows) & 
               Rect(tlB.x, tlB.y, bcols, brows);

    if(debug) {
        cout << "Roi: " << roi << endl;
        cout << "TLA: " << tlA << " TLB: " << tlB << endl;
    }   

    // Bunch of assertions. 
    assert(maskA.type() == CV_8U);
    assert(maskB.type() == CV_8U);
    
    assert(imgA.type() == CV_8UC3);
    assert(imgB.type() == CV_8UC3);

    assert(imgA.cols == maskA.cols);
    assert(imgB.cols == maskB.cols);
    assert(imgA.rows == maskA.rows);
    assert(imgB.rows == maskB.rows);

    //All coordinates in ROI top left

    //Top left corner of first image in ROI space
    int aToRoiX = tlA.x - roi.x;
    int aToRoiY = tlA.y - roi.y;

    //Top left corner of second image in ROI space
    int bToRoiX = tlB.x - roi.x;
    int bToRoiY = tlB.y - roi.y;

    Mat invCost = Mat::zeros(roi.size(), CV_32F);
    Mat path = Mat::zeros(roi.size(), CV_8U);

    //Calculate weighting costs. 
    for(int y = 0; y < roi.height; y++) {
        for(int x = 0; x < roi.width; x++) {
            
            int txA = x - aToRoiX;
            int tyA = y - aToRoiY;
            int txB = x - bToRoiX;
            int tyB = y - bToRoiY;
            
            // Must be satisfied, or else we calculate ROI wrong. 
            assert(tyA >= 0 && tyB >= 0);
            assert(tyA < arows && tyB < brows);
            assert(txA >= 0 && txB >= 0);
            assert(txA < acols && txB < bcols);

            int xA = REMAPX(txA, tyA);
            int yA = REMAPY(txA, tyA);
            int xB = REMAPX(txB, tyB);
            int yB = REMAPY(txB, tyB);

            if(maskA.at<uchar>(yA, xA) == 0 ||
               maskB.at<uchar>(yB, xB) == 0) {
                invCost.at<uchar>(y, x) = 0;
            } else {
                auto va = imgA.at<cv::Vec3b>(yA, xA);
                auto vb = imgB.at<cv::Vec3b>(yB, xB);

                float db = (float)va[0] - (float)vb[0];
                float dr = (float)va[1] - (float)vb[1];
                float dg = (float)va[2] - (float)vb[2];
                
                invCost.at<float>(y, x) = 255 - sqrt(db * db + dr * dr + dg * dg);
            }
            path.at<uchar>(y, x) = 0;
        }
    }

    if(debug) {
        //imwrite("dbg/" + ToString(id) + "_cost_inv.jpg", invCost);
    }
    
    //Calculate path. 
    for(int y = 1; y < roi.height; y++) {
        for(int x = 0; x < roi.width; x++) {
            int min = 0; 

            for(int q = -1; q <= 1; q++) {
                if(x + q < roi.width && x + q >= 0) {
                    if(invCost.at<float>(y - 1, x + q) > 
                       invCost.at<float>(y - 1, x + min)) {
                        min = q;
                    }
                }
            }

            path.at<uchar>(y, x) = (min + 1);
            invCost.at<float>(y, x) += invCost.at<float>(y - 1, x + min);
        }
    }
   
    if(debug) { 
       // imwrite("dbg/" + ToString(id) + "_path.jpg", path * 100);
    }
    //Find best path
    int start = 0;

    for(int x = 1; x < roi.width; x++) {
        if(invCost.at<float>(roi.height - 1, x) > 
           invCost.at<float>(roi.height - 1, start)) {
            start = x;
        }
    }

    //Trace path and update masks
    Mat leftMask = maskA;
    Mat rightMask = maskB;
    auto leftToRoiX = aToRoiX;
    auto rightToRoiX = bToRoiX;
    auto leftCols = acols;

    if(tlA.x > tlB.x) {
        std::swap(leftMask, rightMask);
        std::swap(leftToRoiX, rightToRoiX);
        leftCols = bcols;
    }

    int x = start;

    for(int y = roi.height - 1; y >= 0; y--) {
        for(int q = std::min<int>(x - leftToRoiX + 1 + overlap, leftCols); q < leftCols; q++) {
            //Left mask is black right of path.
            leftMask.at<uchar>(REMAPX(y, q), REMAPY(y, q)) = 0;
        }
        for(int q = 0; std::max<int>(q < x - rightToRoiX - overlap, 0); q++) {
            //Right mask is black left of path
            rightMask.at<uchar>(REMAPX(y, q), REMAPY(y, q)) = 0;
        }
            
        //imgA.at<Vec3b>(remap(y - aToRoiY, x - aToRoiX)) = Vec3b(0, 0, 255);
        //imgB.at<Vec3b>(remap(y - bToRoiY, x - bToRoiX)) = Vec3b(0, 0, 255);

        if(y != 0) {
            int dir = ((int)path.at<uchar>(y, x)) - 1;
            
            x += dir;

            if(debug) {
                cout << "x: " << x << endl;
            }
            assert(x >= 0);
            assert(x < roi.width);
        }   
    }   
   
    if(debug) { 
        //imwrite("dbg/" + ToString(id) + "_a.jpg", imgA);
        //imwrite("dbg/" + ToString(id) + "_b.jpg", imgB);
        imwrite("dbg/" + ToString(id) + "_ma.jpg", maskA);
        //imwrite("dbg/" + ToString(id) + "_mb.jpg", maskB);
    }

    seamTimer.Tick("Image Seamed");
}
template void DynamicSeamer::Find<true>(Mat& imgA, Mat &imgB, Mat &maskA, Mat &maskB, 
        const Point &tlAIn, const Point &tlBIn, int overlap, int id);
template void DynamicSeamer::Find<false>(Mat& imgA, Mat &imgB, Mat &maskA, 
        Mat &maskB, const Point &tlAIn, const Point &tlBIn, int overlap, int id);
}
