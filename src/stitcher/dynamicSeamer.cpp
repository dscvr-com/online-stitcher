#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <vector>

#include "../math/support.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../common/support.hpp"
#include "../common/static_timer.hpp"
#include "dynamicSeamer.hpp"

// Coordinate remap macros, so we can use the same code for vertical and horizontal seaming. 
#define REMAPX(x,y)  (vertical ? x : y)
#define REMAPY(x,y)  (vertical ? y : x)

using namespace std;
using namespace cv;

namespace optonaut {

int DynamicSeamer::debugId = 0;

template <bool vertical>
void DynamicSeamer::Find(Mat& imgA, Mat &imgB, Mat &maskA, Mat &maskB, 
        const Point &tlAIn, const Point &tlBIn, int border, int overlap, int id)
{
    STimer seamTimer;
    static const bool debug = false;
    AssertFalseInProduction(debug);
    static const bool assertsInLoopsOn = false;
    AssertFalseInProduction(assertsInLoopsOn);

    Point tlA;
    Point tlB;
    int acols, bcols, arows, brows;

    // Setup coordinates depending on vertical/horizontal. 
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
        cout << "ROI: " << roi << endl;
    }

    // Margin larger than overlapping image area. We can't handle that. 
    if(roi.width <= border * 2) { 
        return;
    }

    if(debug) {
        cout << "Roi: " << roi << endl;
        cout << "TLA: " << tlA << " TLB: " << tlB << endl;
    }   

    // Some assertions to be on the safe side.  
    Assert(maskA.type() == CV_8U);
    Assert(maskB.type() == CV_8U);
    
    Assert(imgA.type() == CV_8UC3);
    Assert(imgB.type() == CV_8UC3);

    Assert(imgA.cols == maskA.cols);
    Assert(imgB.cols == maskB.cols);
    Assert(imgA.rows == maskA.rows);
    Assert(imgB.rows == maskB.rows);

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
        for(int x = border; x < roi.width - border; x++) {
            
            int txA = x - aToRoiX;
            int tyA = y - aToRoiY;
            int txB = x - bToRoiX;
            int tyB = y - bToRoiY;
           
            if(assertsInLoopsOn) { 
                // ej: Made it possible to disable asserts in thight loops. They cost
                // a lot of performance otherwise!
                // Must be satisfied, or else we calculate ROI wrong. 
                AssertGE(tyA, 0);
                AssertGE(tyB, 0);
                
                Assert(tyA < arows && tyB < brows);
                Assert(txA >= 0 && txB >= 0);
                Assert(txA < acols && txB < bcols);
            }

            int xA = REMAPX(txA, tyA);
            int yA = REMAPY(txA, tyA);
            int xB = REMAPX(txB, tyB);
            int yB = REMAPY(txB, tyB);

            if(maskA.at<uchar>(yA, xA) == 0 ||
               maskB.at<uchar>(yB, xB) == 0) {
                invCost.at<uchar>(y, x) = 0;
            } else {
                invCost.at<float>(y, x) = 
                    255 - sqrt(LeastSquares<cv::Vec3b>::Calculate(imgA, imgB, xA, yA, xB, yB));
            }
            path.at<uchar>(y, x) = 0;
        }
    }

    if(debug) {
       cout << "Writing cost " << id << endl;
       imwrite("dbg/" + ToString(id) + "_cost_inv.jpg", invCost);
    }
    
    // Calculate all paths, from top to bottom.  
    for(int y = 1; y < roi.height; y++) {
        for(int x = border; x < roi.width - border; x++) {
            int min = 0; 

            for(int q = -1; q <= 1; q++) {
                if(x + q < roi.width - border && x + q >= border) {
                    if(invCost.at<float>(y - 1, x + q) >
                       invCost.at<float>(y - 1, x + min)) {
                        min = q;
                    }
                }
            }

            if(assertsInLoopsOn) {
                AssertGE(min + x, border);
                AssertGE(roi.width - border, min + x);
            }
            path.at<uchar>(y, x) = (min + 1);
            invCost.at<float>(y, x) += invCost.at<float>(y - 1, x + min);
        }
    }
   
    if(debug) { 
       cout << "Writing path " << id << endl;
       imwrite("dbg/" + ToString(id) + "_path.jpg", path * 100);
    }
    
    // Start at the bottom, find the best bath.  
    int start = 1 + border;

    for(int x = 1 + border; x < roi.width - border; x++) {
        if(invCost.at<float>(roi.height - 1, x) > 
           invCost.at<float>(roi.height - 1, start)) {
            start = x;
        }
    }

    //Trace path from bottom to top and update masks
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

    if(debug) {
        cout << "start: " << x << endl;
    }

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

            if(assertsInLoopsOn) {
                AssertGE(x, border);
                AssertGE(roi.width - border, x);
            }
        }   
    }   
   
    if(debug) { 
        imwrite("dbg/" + ToString(id) + "_ma.jpg", maskA);
    }
    
    seamTimer.Tick("Image Seamed");
}
template void DynamicSeamer::Find<true>(Mat& imgA, Mat &imgB, Mat &maskA, Mat &maskB, 
        const Point &tlAIn, const Point &tlBIn, int border, int overlap, int id);
template void DynamicSeamer::Find<false>(Mat& imgA, Mat &imgB, Mat &maskA, 
        Mat &maskB, const Point &tlAIn, const Point &tlBIn, int border, int overlap, int id);
}
