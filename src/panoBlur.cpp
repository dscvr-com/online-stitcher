#include <iostream>
#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>

#include "imgproc/panoramaBlur.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

void printUsage() {
}

int main(int argc, char** argv) {

    if(argc < 2) {
        printUsage();
        return 1;
    }

    Mat in = imread(argv[1]);

    if(in.cols == 0 || in.rows == 0) {
        cout << "Cannot read input image." << endl;
        return 1;
    }

    Mat res;

    PanoramaBlur blur(in.size(), Size(in.cols, in.cols / 2));
    blur.Blur(in, res);

    imwrite(argv[2], res);
    
    return 0;
}
