
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <vector>
#include "support.hpp"

#ifndef OPTONAUT_VERTICAL_DP_SEAMER_HEADER
#define OPTONAUT_VERTICAL_DP_SEAMER_HEADER

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {
class VerticalDynamicSeamer 
{
public:
    static void Find(Mat& imageA, Mat &imageP, Mat &maskA, Mat &maskB, const Point &tlA, const Point &tlB, int overlap = 0, int id = 0);
};

void VerticalDynamicSeamer::Find(Mat& imgA, Mat &imgB, Mat &maskA, Mat &maskB, const Point &tlA, const Point &tlB, int overlap, int id)
{
    static const bool debug = false;

    Rect roi = Rect(tlA.x, tlA.y, imgA.cols, imgA.rows) & 
               Rect(tlB.x, tlB.y, imgB.cols, imgB.rows);

    if(debug) {
        cout << "Roi: " << roi << endl;
        cout << "TLA: " << tlA << " TLB: " << tlB << endl;
    }   

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
    for(int x = 0; x < roi.width; x++) {
        for(int y = 0; y < roi.height; y++) {
            
            // Must be satisfied, or else we calculate ROI wrong. 
            assert(y - aToRoiY >= 0 && y - bToRoiY >= 0);
            assert(y - aToRoiY < maskA.rows && y - bToRoiY < maskB.rows);
            assert(x - aToRoiX >= 0 && x - bToRoiX >= 0);
            assert(x - aToRoiX < maskA.cols && x - bToRoiX < maskB.cols);

            if(maskA.at<uchar>(y - aToRoiY, x - aToRoiX) == 0 ||
               maskB.at<uchar>(y - bToRoiY, x - bToRoiX) == 0) {
                invCost.at<uchar>(y, x) = 0;
            } else {
                auto va = imgA.at<cv::Vec3b>(y - aToRoiY, x - aToRoiX);
                auto vb = imgB.at<cv::Vec3b>(y - bToRoiY, x - bToRoiX);

                auto db = (float)va[0] - (float)vb[0];
                auto dr = (float)va[1] - (float)vb[1];
                auto dg = (float)va[2] - (float)vb[2];
                
                invCost.at<float>(y, x) = 255 - sqrt(db * db + dr * dr + dg * dg);
            }
            path.at<uchar>(y, x) = 0;
        }
    }

    if(debug) {
        imwrite("dbg/" + ToString(id) + "_cost_inv.jpg", invCost);
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
        imwrite("dbg/" + ToString(id) + "_path.jpg", path * 100);
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
    Mat &leftMask = maskA;
    Mat &rightMask = maskB;
    auto leftToRoiX = aToRoiX;
    auto rightToRoiX = bToRoiX;

    if(tlA.x > tlB.x) {
        leftMask = maskB;
        rightMask = maskA;
        leftToRoiX = bToRoiX;
        rightToRoiX = aToRoiX;
    }

    int x = start;

    for(int y = roi.height - 1; y >= 0; y--) {
        for(int q = std::min<int>(x - leftToRoiX + 1 + overlap, leftMask.cols); q < leftMask.cols; q++) {
            //Left mask is black right of path.
            leftMask.at<uchar>(y, q) = 0;
        }
        for(int q = 0; std::max<int>(q < x - rightToRoiX - overlap, 0); q++) {
            //Right mask is black left of path
            rightMask.at<uchar>(y, q) = 0;
        }
            
        //imgA.at<Vec3b>(y - aToRoiY, x - aToRoiX) = Vec3b(0, 0, 255);
        //imgB.at<Vec3b>(y - bToRoiY, x - bToRoiX) = Vec3b(0, 0, 255);

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
        imwrite("dbg/" + ToString(id) + "_a.jpg", imgA);
        imwrite("dbg/" + ToString(id) + "_b.jpg", imgB);
        imwrite("dbg/" + ToString(id) + "_ma.jpg", maskA);
        imwrite("dbg/" + ToString(id) + "_mb.jpg", maskB);
    }
};
}
#endif
