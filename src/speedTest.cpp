#include <iostream>
#include <opencv2/core.hpp>

#include "common/static_timer.hpp"
#include "common/logger.hpp"
#include "math/support.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

int main(int, char**) {
    
    int w = 300;
    int h = 300;

    #define COUNT 1000

    Mat img(w, h, CV_32FC2, Scalar(1.f, 1.f));
    Mat out1 = Mat::zeros(w, h, CV_32FC2);
    Mat out2 = Mat::zeros(w, h, CV_32FC2);

    STimer timer; 

    for(int i = 0; i < COUNT; i++) { 
        for(int x = 0; x < w; x++) {
            for(int y = 0; y < h; y++) {
                Vec2f v = img.at<Vec2f>(y, x);
                out1.at<Vec2f>(y, x) = v * ((float)x / (float)(y + 1));
            }
        }
    }

    timer.Tick("Vec Interface");
    
    for(int i = 0; i < COUNT; i++) { 
        float* pImg = (float*)img.ptr();
        float* pOut = (float*)out2.ptr();

        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                float a = *pImg++;
                float b = *pImg++;

                float fac = ((float)x / (float)(y + 1));

                *pOut++ = a * fac;
                *pOut++ = b * fac;
            }
        }
    }
    
    AssertMatEQ<Vec2f>(out1, out2);
    
    timer.Tick("Raw pointers");

    for(int i = 0; i < COUNT; i++) { 
        out1 += img;
    }
    
    timer.Tick("CV Add Mat");
    
    for(int i = 0; i < COUNT; i++) { 
        float* pImg = (float*)img.ptr();
        float* pOut = (float*)out2.ptr();

        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                *pOut = *pOut + *pImg++;
                pOut++;
                *pOut = *pOut + *pImg++;
                pOut++;
            }
        }
    }
    
    timer.Tick("Raw pointers");

    AssertMatEQ<Vec2f>(out1, out2);


    return 0;
}
