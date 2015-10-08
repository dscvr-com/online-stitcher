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

    static const bool isAsync = true;

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
        auto image = InputImageFromFile(files[i]);
        
        image->intrinsics = iPhone5Intrinsics;
        
        //Create stack-local ref to mat. Clear image mat.
        //This is to simulate hard memory management.
        Mat tmpMat = image->image.data;
        
        image->image = Image(Mat(0, 0, CV_8UC3));
        
        image->dataRef.data = tmpMat.data;
        image->dataRef.width = tmpMat.cols;
        image->dataRef.height = tmpMat.rows;
        image->dataRef.colorSpace = colorspace::RGB;

        if(i == 0) {
            pipe = shared_ptr<Pipeline>(new Pipeline(Pipeline::iosBase, Pipeline::iosZero, image->intrinsics, "tmp/", RecorderGraph::ModeTruncated, isAsync));

            Pipeline::debug = true;
        }

        pipe->Push(image);
        tmpMat.release();

        if(isAsync) {
            auto now = system_clock::now(); 
            auto diff = now - lt;
            auto sleep = 30ms - diff;
            //cout << "Sleeping for " << duration_cast<microseconds>(sleep).count() << endl;
            this_thread::sleep_for(sleep);
        }
    }

    pipe->Finish();
    
    if(pipe->HasResults()) {
        {
            auto left = pipe->FinishLeft();
            imwrite("dbg/left.jpg", left->image.data);
            left->image.Unload();  
            left->mask.Unload();  
        }
        {
            auto right = pipe->FinishRight();
            imwrite("dbg/right.jpg", right->image.data);    
            right->image.Unload();  
            right->mask.Unload();  
        }
    } else {
        cout << "No results." << endl;
    }
    
    if(Pipeline::debug) {
        //auto alignedDebug = pipe->FinishAlignedDebug();
        //imwrite("dbg/aligned-debug.jpg", alignedDebug->image);
        auto aligned = pipe->FinishAligned();
        imwrite("dbg/aligned.jpg", aligned->image.data);
    }
    
    pipe->Dispose();

    return 0;
}
