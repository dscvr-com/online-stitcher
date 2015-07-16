#include "wrapper.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;

template <typename T>
void bufferFromStringFile(T buf[], int len, string file) {
    ifstream input(file);

    for(int i = 0; i < len; i++) {
        input >> buf[i];
    }
}

void bufferFromBinFile(unsigned char buf[], int *len, string file) {

    ifstream input(file, std::ios::binary);

    int i;
    for(i = 0; i < *len && input.good(); i++) {
        buf[i] = input.get();
    }
    *len = i;
}

int main(int argc, char* argv[]) {
    double extrinsics[9];
    double intrinsics[9];
    double outExtrinsics[16];
    int width = 640; 
    int height = 480;

    assert(argc == 2);

    string path(argv[1]);

    unsigned char image[640*480*4];

    int dataSize = 640*480*4;
    bufferFromBinFile(image, &dataSize, path + string("/data.bin"));
    cout << "Read " << dataSize << " bytes of image" << endl;
    bufferFromStringFile<double>(extrinsics, 9, path + string("/extrinsics.txt"));
    bufferFromStringFile<double>(intrinsics, 9, path + string("/intrinsics.txt"));

    optonaut::Push(extrinsics, intrinsics, image, width, height, outExtrinsics, 0);
    
    for(int i = 0; i < 16; i++) {
        cout << outExtrinsics[i] << " ";
    }
    cout << endl;

    optonaut::Push(extrinsics, intrinsics, image, width, height, outExtrinsics, 1);

    for(int i = 0; i < 16; i++) {
        cout << outExtrinsics[i] << " ";
    }
    cout << endl;
}