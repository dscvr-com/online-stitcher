//
//  recorder.c
//  Optonaut
//
//  Created by Emi on 29/08/15.
//  Copyright © 2015 Optonaut. All rights reserved.
//

#include "recorder.hpp"

using namespace cv;
using namespace std;

namespace optonaut {
    //Portrait to landscape (use with ios app)
    double iosBaseData[16] = {
        1, 0, 0, 0,
        0, -1, 0, 0,	
        0, 0, -1, 0,
        0, 0, 0, 1
    };

    //Landscape L to R (use with android app)
    double androidBaseData[16] = {
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    //Base picked from exsiting data - we might find something better here.
    double iosZeroData[16] = {
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 0,
        0, 0, 0, 1
    };

    //Base picked from exsiting data - we might find something better here.
    double androidZeroData[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };


    Mat Recorder::androidBase(4, 4, CV_64F, androidBaseData);
    Mat Recorder::iosBase(4, 4, CV_64F, iosBaseData);
    Mat Recorder::iosZero = Recorder::iosBase * Mat(4, 4, CV_64F, iosZeroData) * Recorder::iosBase.inv();
    Mat Recorder::androidZero = Recorder::androidBase * Mat(4, 4, CV_64F, androidZeroData) * Recorder::androidBase.inv();

    string Recorder::tempDirectory = "tmp/";
    string Recorder::version = "0.7.0";
    
    bool Recorder::exposureEnabled = false;
    bool Recorder::alignmentEnabled = true;
}

