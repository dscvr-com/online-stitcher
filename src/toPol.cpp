#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printUsage() {
   cout << "Simple optograph to little-world converter." << endl;
   cout << "usage: to-polar [INPUT-IMAGE] [OUTPUT-IMAGE]" << endl; 
}

int main(int argc, char** argv) {

    if(argc != 3) {
        printUsage();
        return 0;
    }
    assert(argc == 3);

    Mat pol;
    Mat in = imread(argv[1]);

    if(in.cols == 0 || in.rows == 0) {
        cout << "Cannot read input image." << endl;
        return 1;
    }

    // Old quadratic Optograph format, we need to cut black borders away. 
    if(in.cols == in.rows) {
        in = in(Rect(0, 630, in.cols, in.rows - 630 * 2)); 
    }

    // Make our input panorama square
    resize(in, in, Size(in.rows, in.rows));

    // Orient our input panorama so that the floor will be in 
    // the center of the output. 
    in = in.t();
    flip(in, in, -1);

    // Inverse log transform. Center around input center. Set radius so 
    // result contains all of the input image. 
    linearPolar(in, pol, 
            Point2f(in.cols / 2, in.rows / 2), (in.rows / 2) * sqrt(2), 
            CV_WARP_FILL_OUTLIERS | CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);

    imwrite(argv[2], pol);
    
    return 0;
}
