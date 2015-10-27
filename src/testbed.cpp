#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "intrinsics.hpp"
#include "recorder.hpp"
#include "stitcher.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}


void Record(vector<string> &files, CheckpointStore &leftStore, CheckpointStore &rightStore) {

    namedWindow("Recorder", WINDOW_AUTOSIZE);

    if(files.size() == 0) {
        cout << "No Input." << endl;
        return;
    }

    static const bool isAsync = true;
    shared_ptr<Recorder> recorder(NULL);

    for(size_t i = 0; i < files.size(); i++) {
        auto lt = system_clock::now();
        auto image = InputImageFromFile(files[i], false);
            
        image->intrinsics = iPhone5Intrinsics;
        
        //Create stack-local ref to mat. Clear image mat.
        //This is to simulate hard memory management.
        Mat tmpMat = image->image.data;

        //imshow("Recorder", tmpMat);
        //waitKey(0); 
        
        image->image = Image(Mat(0, 0, CV_8UC4));
        
        image->dataRef.data = tmpMat.data;
        image->dataRef.width = tmpMat.cols;
        image->dataRef.height = tmpMat.rows;
        image->dataRef.colorSpace = colorspace::RGB;

        if(i == 0) {
            recorder = shared_ptr<Recorder>(new Recorder(Recorder::iosBase, Recorder::iosZero, image->intrinsics, leftStore, rightStore, RecorderGraph::ModeTruncated, isAsync));
        }

        recorder->Push(image);
        tmpMat.release();

        if(isAsync) {
            auto now = system_clock::now(); 
            auto diff = now - lt;
            auto sleep = 10ms - diff;
            //cout << "Sleeping for " << duration_cast<microseconds>(sleep).count() << endl;
            this_thread::sleep_for(sleep);
        }
    }

    destroyWindow("Recorder");

    recorder->Finish();
    recorder->Dispose();
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    CheckpointStore leftStore("tmp/left/", "tmp/shared/");
    CheckpointStore rightStore("tmp/right/", "tmp/shared/");

    cout << "Starting." << endl;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);

    if(!leftStore.HasUnstitchedRecording()) {
        cout << "Recording." << endl;
        Record(files, leftStore, rightStore);
    }

    if(!leftStore.HasUnstitchedRecording()) {
        cout << "No results." << endl;
        return 0;
    }
    
    cout << "Create callbacks." << endl;

    ProgressCallback progress([](float) -> bool {
                //cout << (int)(progress * 100) << "% ";
                return true;
            });
    ProgressCallbackAccumulator callbacks(progress, {0.5, 0.5});

    {
        cout << "Start left stitcher." << endl;
        Stitcher leftStitcher(leftStore);
        auto left = leftStitcher.Finish(callbacks.At(0), false, "dbg/left");
        imwrite("dbg/left.jpg", left->image.data);
        left->image.Unload();  
        left->mask.Unload();  
    }
    {
        cout << "Start right stitcher." << endl;
        Stitcher rightStitcher(rightStore);
        auto right = rightStitcher.Finish(callbacks.At(1), false, "dbg/right");
        imwrite("dbg/right.jpg", right->image.data);    
        right->image.Unload();  
        right->mask.Unload();  
    }

    //leftStore.Clear();
    //rightStore.Clear();
    
    return 0;
}
