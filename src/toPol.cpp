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

inline float scaleX(float x, float width) {
   static const float scale = 1;
  // static const float scaleSqrt = sqrt(1 / scale);
   return width * sqrt(x * scale / width);
}

inline void createLittleWorld(const Mat &optograph, Mat &pol, const int targetWidth = -1, const int targetHeight = -1) {
    // Old quadratic Optograph format, we need to cut black borders away. 
   
    Mat in = optograph;

    if(in.cols == in.rows) {
        in = in(Rect(0, 630, in.cols, in.rows - 630 * 2)); 
    }

    // 1) Make our input panorama square.
    // 2) Orient our input panorama so that the floor will be in.
    // the center of the output. 
    // 3) Non-linearly scale image - make the sky bigger.
    const int width = in.cols;
    const int height = in.rows;
    int rWidth = 0;
    int rHeight = 0;
    if(targetWidth == -1 || targetHeight == -1) {
        rWidth = min(height, width);
        rHeight = rWidth;
    } else {
        rWidth = max(targetHeight, targetWidth) * 3;
        rHeight = rWidth;
    }
    const float widthRatio = (float)width / (float)rHeight;
    const float heightRatio = (float)height / (float)rWidth;
    
    Mat mapx(rHeight, rWidth, CV_32F);
    Mat mapy(rHeight, rWidth, CV_32F);

    for(int y = 0; y < rHeight; y++) {
        for(int x = 0; x < rWidth; x++) {
            mapx.at<float>(y, x) = y * widthRatio;
            mapy.at<float>(y, x) = (rWidth - scaleX(x, rWidth)) * heightRatio;
        }
    }

    Mat resized(rHeight, rWidth, in.type());
    remap(in, resized, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    double destRadius = (rWidth / 2) * sqrt(2);

    // Inverse linear polar transform. Center around input center. Set radius so 
    // result contains all of the input image. 
    linearPolar(resized, pol, 
            Point2f(rWidth / 2, rHeight / 2), destRadius, 
            CV_WARP_FILL_OUTLIERS | CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);

    const int maxDimension = max(targetWidth, targetHeight);
    resize(pol, pol, Size(maxDimension, maxDimension));

    pol = pol(
            Rect((maxDimension - targetWidth) / 2, 
                (maxDimension - targetHeight) / 2, 
                targetWidth, 
                targetHeight)
            );
}

int main(int argc, char** argv) {

    if(argc != 3) {
        printUsage();
        return 0;
    }
    Assert(argc == 3);

    Mat in = imread(argv[1]);

    if(in.cols == 0 || in.rows == 0) {
        cout << "Cannot read input image." << endl;
        return 1;
    }
    Mat pol;
    createLittleWorld(in, pol, 1200, 630);
    imwrite(argv[2], pol);
    
    return 0;
}
