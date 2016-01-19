#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>

#include "math/projection.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

void printUsage() {
   cout << "Simple equiretangular optograph to cubemap converter." << endl;
   cout << "usage: to-cube-map [INPUT-IMAGE] [OUTPUT-IMAGE] [WIDTH] [FACE-ID] [SUB X] [SUB Y] [SUB WIDTH] [SUB HEIGHT]" << endl; 
}

int main(int argc, char** argv) {

    if(argc < 3) {
        printUsage();
        return 1;
    }

    Mat in = imread(argv[1]);

    if(in.cols == 0 || in.rows == 0) {
        cout << "Cannot read input image." << endl;
        return 1;
    }

    int w = 1024;

    if(argc >= 4) {
        w = std::stoi(string(argv[3]));
    }
   
    if(argc >= 5) { 
        int faceId = std::stoi(string(argv[4]));

        float subX = 0;
        float subY = 0;
        float subW = 1;
        float subH = 1;

        if(faceId < 0 || faceId > 5) {
            cout << "Faceid has to be between 0 and 5, inclusive." << endl;
            return 1;
        }

        if(argc == 9) {
            subX = std::atof(argv[5]);
            subY = std::atof(argv[6]);
            subW = std::atof(argv[7]);
            subH = std::atof(argv[8]);
        }

        Mat face;
        CreateCubeMapFace(in, face, faceId, w, w, subX, subY, subW, subH);
        imwrite(argv[2], face);
    } else {
        Mat target = Mat::zeros(w * 3, w * 4, CV_8UC3); 

        for(int i = 0; i < 4; i++) {
            Mat face = target(Rect(w * i, w, w, w));
            CreateCubeMapFace(in, face, i, w, w);
        }
        Mat face = target(Rect(w * 3, 0, w, w));
        CreateCubeMapFace(in, face, 4, w, w);
        
        face = target(Rect(w * 3, w * 2, w, w));
        CreateCubeMapFace(in, face, 5, w, w);
        
        imwrite(argv[2], target);
    }
    
    return 0;
}
