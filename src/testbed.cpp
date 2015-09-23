#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>

#include "intrinsics.hpp"
#include "pipeline.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char* argv[]) {

    static const bool isAsync = false;

    int n = argc - 1;
    shared_ptr<Pipeline> pipe(NULL);
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);
 

    for(int i = 0; i < n; i++) {
        auto lt = system_clock::now();
        auto image = ImageFromFile(files[i]);
        
        image->intrinsics = iPhone6Intrinsics;
        
        //Create stack-local ref to mat. Clear image mat.
        //This is to simulate hard memory management.
        Mat tmpMat = image->img;
        
        image->img = Mat(0, 0, CV_8UC3);
        
        image->dataRef.data = tmpMat.data;
        image->dataRef.width = tmpMat.cols;
        image->dataRef.height = tmpMat.rows;
        image->dataRef.colorSpace = colorspace::RGB;

        if(i == 0) {
            pipe = shared_ptr<Pipeline>(new Pipeline(Pipeline::iosBase, Pipeline::iosZero, image->intrinsics, RecorderGraph::ModeAll, isAsync));

            Pipeline::debug = true;
        }

        pipe->Push(image);
        tmpMat.release();

        if(isAsync) {
            auto now = system_clock::now(); 
            auto diff = now - lt;
            auto sleep = 30ms - diff;
            cout << "Sleeping for " << duration_cast<microseconds>(sleep).count() << endl;
            this_thread::sleep_for(sleep);
        }
    }

    pipe->Finish();
    
    if(Pipeline::debug) {
        //auto alignedDebug = pipe->FinishAlignedDebug();
        //imwrite("dbg/aligned-debug.jpg", alignedDebug->image);
        auto aligned = pipe->FinishAligned();
        imwrite("dbg/aligned.jpg", aligned->image);
    }
    
    if(pipe->HasResults()) {
        {
            auto left = pipe->FinishLeft();
            imwrite("dbg/left.jpg", left->image);
            left->image.release();  
            left->mask.release();  
        }
        {
            auto right = pipe->FinishRight();
            imwrite("dbg/right.jpg", right->image);    
            right->image.release();  
            right->mask.release();  
        }
    } else {
        cout << "No results." << endl;
    }

    pipe->Dispose();

    return 0;
}
