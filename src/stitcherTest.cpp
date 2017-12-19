#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "minimal/imagePreperation.hpp"
#include "recorder/recorder.hpp"
#include "stitcher/stitcherSink.hpp"
#include "common/intrinsics.hpp"
#include "common/backtrace.hpp"
#include "common/drawing.hpp"
#include "stitcher/stitcher.hpp"
#include "stitcher/globalAlignment.hpp"
#include "io/io.hpp"
#include "recorder/recorder2.hpp"
#include "recorder/multiRingRecorder2.hpp"
#include "recorder/recorderParamInfo.hpp"

// Comment in this define to use the motor pipeline for testing. 
 #define USE_THREE_RING

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

#ifdef USE_THREE_RING
typedef MultiRingRecorder RecorderToUse;
const int RecorderMode = RecorderGraph::ModeTruncated; 
#else
typedef Recorder2 RecorderToUse;
const int RecorderMode = RecorderGraph::ModeCenter; 
#endif
CheckpointStore leftStore("tmp/left/", "tmp/shared/");
CheckpointStore rightStore("tmp/right/", "tmp/shared/");
    
StorageImageSink leftSink(leftStore);
StorageImageSink rightSink(rightStore);

shared_ptr<RecorderGraph> graph;
shared_ptr<RecorderGraph> preGraph;
Mat intrinsics;
Size imageSize;
vector<InputImageP> allImages;
vector<Mat> originalExtrinsics;

void Record(vector<string> &files) {

    if(files.size() == 0) {
        cout << "No Input." << endl;
        return;
    }

    static const bool isAsync = true;
    static const int mode = minimal::ImagePreperation::ModeNone;
    shared_ptr<RecorderToUse> recorder(NULL);
        
    Mat base, zero;

    if(mode == minimal::ImagePreperation::ModeIOS) {
        base = optonaut::Recorder::iosBase;
        zero = Recorder::iosZero;
    } else if(mode == minimal::ImagePreperation::ModeNone) {
        base = Mat::eye(4, 4, CV_64F);
        zero = Mat::eye(4, 4, CV_64F);
    } else {
        base = optonaut::Recorder::androidBase;
        zero = Recorder::androidZero;
    }

    //DebugHook::Instance = &hook;

    for(size_t i = 0; i < files.size(); i++) {
        auto lt = system_clock::now();
        auto image = InputImageFromFile(files[i], false);
        cout << "[Record] Loading input: " << files[i] << endl;
        
        if(mode == minimal::ImagePreperation::ModeNone) {
            // If we deactivate base adjustment, apply a transposition to 
            // the extrinsics, to counter the transposition that 
            // happens inside recorder. 
            image->originalExtrinsics = image->originalExtrinsics.t();
            image->adjustedExtrinsics = image->adjustedExtrinsics.t();
        }
            
        //Create stack-local ref to mat. Clear image mat.
        //This is to simulate hard memory management.
        Mat tmpMat = image->image.data;

        image->image = Image(Mat(0, 0, CV_8UC4));
        
        image->dataRef.data = tmpMat.data;
        image->dataRef.width = tmpMat.cols;
        image->dataRef.height = tmpMat.rows;
        image->dataRef.colorSpace = colorspace::RGB;

        allImages.push_back(image);

        if(i == 0) {

            //std::string debugPath = "dbg/debug_out/";
            std::string debugPath = "";
#ifdef USE_THREE_RING 
            // MotorControlRecorder
            recorder = std::make_shared<RecorderToUse>(base, zero, 
                        image->intrinsics, leftSink, rightSink, RecorderMode, 1.0, "", RecorderParamInfo(0.7, 0.5, 0.55, -0.1, 2.0, true));
#else
            // Recorder2 
            recorder = std::make_shared<RecorderToUse>(base, zero, 
                        image->intrinsics, RecorderMode, 5.0, "");
#endif

            recorder->SetIdle(false);
        }

        recorder->Push(image);

        tmpMat.release();
        
        if(recorder->RecordingIsFinished()) {
            break;
        }

        if(isAsync) {
            auto now = system_clock::now(); 
            auto diff = now - lt;
            auto sleep = chrono::milliseconds(10) - diff;

	    if(sleep.count() > 0) {
            	cout << "Sleeping for " << sleep.count() << endl;

            	this_thread::sleep_for(sleep);
	    }
        }
    }
    
    recorder->Finish();
#ifdef USE_THREE_RING
    // Motor
    optonaut::Stitcher leftStitcher(leftStore);
    optonaut::Stitcher rightStitcher(rightStore);

    auto left = leftStitcher.Finish(ProgressCallback::Empty);
    auto right = rightStitcher.Finish(ProgressCallback::Empty);
#else

    // 1 ring
    auto left = recorder->GetLeftResult();
    auto right = recorder->GetRightResult();
#endif
    
    imwrite("dbg/right.jpg", right->image.data);
    imwrite("dbg/left.jpg", left->image.data);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    RegisterCrashHandler();

    StitcherSink stitcherSink;

    cout << "Starting." << endl;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        //cout << "imageName :" << imageName << endl;
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);

    cout << "Recording." << endl;
    Record(files);
    return 0;
}
