#include <iostream>
#include <algorithm>
#include "io/io.hpp"
#include "debug/visualDebugHook.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);
    
    VisualDebugHook debugger;

    for(int i = 0; i < min(n, 250); i += 1) {

        cout << "Adding image " << i << endl;

        auto img = InputImageFromFile(files[i], false);
        debugger.RegisterImageRotationModel(img->image.data, img->originalExtrinsics, img->intrinsics);

    }
    debugger.Draw();
    debugger.WaitForExit();

    return 0;
}
