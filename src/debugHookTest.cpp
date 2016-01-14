#include <iostream>
#include <algorithm>
#include "io/io.hpp"
#include "debug/visualDebugHook.hpp"
#include "minimal/imagePreperation.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    
    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(
            argc, argv, false, 3, 100);
    VisualDebugHook debugger;

    size_t n = images.size();

    for(size_t i = 0; i < n; i += 1) {

        auto img = images[i];

        cout << "Adding image " << i << endl;

        debugger.RegisterImageRotationModel(img->image.data, img->originalExtrinsics, img->intrinsics);
        debugger.RegisterCamera(img->originalExtrinsics, 0, 0, 0, i);

    }
    debugger.Draw();
    debugger.WaitForExit();

    return 0;
}
