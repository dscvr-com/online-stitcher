#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

#include "pipeline.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char* argv[]) {
    
    int n = argc - 1;
    shared_ptr<Pipeline> pipe(NULL);
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);
    
    for(int i = 0; i < n; i++) {
        auto image = ImageFromFile(files[i]);

        this_thread::sleep_for(0.5s);
        
        if(i == 0) {
            pipe = shared_ptr<Pipeline>(new Pipeline(Pipeline::iosBase, Pipeline::iosZero, image->intrinsics));
        }

        pipe->Push(image);
    }
    
    if(pipe->HasResults()) {
        auto left = pipe->FinishLeft();
        auto right = pipe->FinishRight();
            
        imwrite("dbg/left.jpg", left->image);    
        imwrite("dbg/right.jpg", right->image);    
    } else {
        cout << "No results." << endl;
    }

    pipe->Dispose();

    return 0;
}
