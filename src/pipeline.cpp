//
//  pipeline.c
//  Optonaut
//
//  Created by Emi on 29/08/15.
//  Copyright © 2015 Optonaut. All rights reserved.
//

#include "pipeline.hpp"

using namespace cv;
using namespace std;

namespace optonaut {
    //Portrait to landscape (use with ios app)
    double iosBaseData[16] = {
        -1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
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


    Mat Pipeline::androidBase(4, 4, CV_64F, androidBaseData);
    Mat Pipeline::iosBase(4, 4, CV_64F, iosBaseData);
    Mat Pipeline::iosZero = Pipeline::iosBase * Mat(4, 4, CV_64F, iosZeroData) * Pipeline::iosBase.inv();

    string Pipeline::tempDirectory = "tmp/";
    string Pipeline::version = "0.5.0";
    bool Pipeline::debug = false;
}

