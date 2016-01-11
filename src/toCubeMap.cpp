#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printUsage() {
   cout << "Simple equiretangular optograph to cubemap converter." << endl;
   cout << "usage: to-cube-map [INPUT-IMAGE] [OUTPUT-IMAGE] [FACE-ID] [SUB X] [SUB Y] [SUB WIDTH] [SUB HEIGHT]" << endl; 
}

float faceTransform[6][2] = 
    { 
        {0, 0},
        {M_PI / 2, 0},
        {M_PI, 0},
        {-M_PI / 2, 0},
        {0, -M_PI / 2},
        {0, M_PI / 2}
    };


inline void createCubeMapFace(const Mat &optograph, Mat &face, 
        const int faceId, const int width, const int height, 
        const float subX = 0, const float subY = 0, 
        const float subW = 1, const float subH = 1) {

    assert(faceId >= 0 && faceId < 6);
    assert(subX >= 0 && subX + subW <= 1);
    assert(subY >= 0 && subY + subH <= 1);
    assert(subW > 0);
    assert(subH > 0);

    Mat in = optograph;
    float inWidth;
    float inHeight;
    float inOffsetTop;
    float inOffsetLeft;

    if(in.cols == in.rows) {
        // Old quadratic Optograph format, we need to cut black borders away. 
        inWidth = in.cols;
        inHeight = in.rows;
        inOffsetTop = 0;
        inOffsetLeft = 0;
    } else {
        // Optimised format - fake black border 
        inWidth = in.cols;
        inHeight = inWidth / 2;
        inOffsetTop = (inHeight - in.rows) / 2;
        inOffsetLeft = 0;
    }

    Mat mapx(height, width, CV_32F);
    Mat mapy(height, width, CV_32F);

    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {

            // Map texture to unit space [0, 1]
            float nx = (float)y / (float)height;
            float ny = (float)x / (float)width;

            // Subface
            nx *= subW;
            ny *= subH;

            nx += subX;
            ny += subY;

            // Remap to [-an, an]
            nx -= 0.5f;
            ny -= 0.5f;

            nx *= 2 * an; 
            ny *= 2 * an; 

            float u, v;

            // Project
            if(ftv == 0) {
                // Center faces
                u = atan2(nx, ak);
                v = atan2(ny * cos(u), ak);
                u += ftu; 
            } else if(ftv > 0) { 
                // Bottom face 
                float d = sqrt(nx * nx + ny * ny);
                v = M_PI / 2 - atan2(d, ak);
                u = atan2(ny, nx);
            } else {
                // Top face
                float d = sqrt(nx * nx + ny * ny);
                v = -M_PI / 2 + atan2(d, ak);
                u = atan2(-ny, nx);
            }
            u = u / (M_PI); 
            v = v / (M_PI / 2);

            // Warp around
            while (v < -1) {
                v += 2;
                u += 1;
            } 
            while (v > 1) {
                v -= 2;
                u += 1;
            } 

            while(u < -1) {
                u += 2;
            }
            while(u > 1) {
                u -= 2;
            }

            // Map to texture sampling space
            u = u / 2.0f + 0.5f;
            v = v / 2.0f + 0.5f;

            u = u * (inWidth - 1) - inOffsetLeft;
            v = v * (inHeight - 1) - inOffsetTop;

            // Save in map
            mapx.at<float>(x, y) = u;
            mapy.at<float>(x, y) = v; 
        }
    }

    if(face.cols != width || face.rows != height || face.type() != in.type()) {
        face = Mat(width, height, in.type());
    }
    remap(in, face, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
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

    const int w = 1024;
   
    if(argc >= 4) { 
        int faceId = std::stoi(string(argv[3]));

        float subX = 0;
        float subY = 0;
        float subW = 1;
        float subH = 1;

        if(faceId < 0 || faceId > 5) {
            cout << "Faceid has to be between 0 and 5, inclusive." << endl;
            return 1;
        }

        if(argc == 8) {
            subX = std::atof(argv[4]);
            subY = std::atof(argv[5]);
            subW = std::atof(argv[6]);
            subH = std::atof(argv[7]);
        }

        Mat face;
        createCubeMapFace(in, face, faceId, w, w, subX, subY, subW, subH);
        imwrite(argv[2], face);
    } else {
        Mat target = Mat::zeros(w * 3, w * 4, CV_8UC3); 

        for(int i = 0; i < 4; i++) {
            Mat face = target(Rect(w * i, w, w, w));
            createCubeMapFace(in, face, i, w, w);
        }
        Mat face = target(Rect(w * 3, 0, w, w));
        createCubeMapFace(in, face, 4, w, w);
        
        face = target(Rect(w * 3, w * 2, w, w));
        createCubeMapFace(in, face, 5, w, w);
        
        imwrite(argv[2], target);
    }
    
    return 0;
}
