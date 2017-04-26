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
#include "stitcher/convertToStereo.hpp"
#include "io/io.hpp"
#include "recorder/recorder2.hpp"
#include "recorder/motorControlRecorder.hpp"

// Comment in this define to use the motor pipeline for testing. 
// #define USE_MOTOR

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

#ifdef USE_MOTOR
typedef MotorControlRecorder RecorderToUse;
const int RecorderMode = RecorderGraph::ModeTruncated; 
#else
typedef Recorder2 RecorderToUse;
const int RecorderMode = RecorderGraph::ModeCenter; 
#endif
CheckpointStore outStore("tmp/out/", "tmp/shared");
CheckpointStore leftStore("tmp/left/", "tmp/shared/");
CheckpointStore rightStore("tmp/right/", "tmp/shared/");
    
StorageImageSink recorderOut(outStore);
//const int RecorderMode = RecorderGraph::ModeCenter; 

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
            
        //image->intrinsics = iPhone6Intrinsics;
        
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

        allImages.push_back(image);

        if(i == 0) {

            //std::string debugPath = "dbg/debug_out/";
            std::string debugPath = "";
#ifdef USE_MOTOR
            // MotorControlRecorder
            recorder = std::make_shared<RecorderToUse>(base, zero, 
                        image->intrinsics, recorderOut, RecorderMode, 8, "");
#else
            // Recorder2 
            // TODO: Change tolerance back!
            recorder = std::make_shared<RecorderToUse>(base, zero, 
                        image->intrinsics, RecorderMode, 16, "");
#endif

            recorder->SetIdle(false);

            // Needed for debug. 
            // intrinsics = image->intrinsics;
            // imageSize = Size(intrinsics.at<double>(0, 2), 
            //        intrinsics.at<double>(1, 2));
           // graph = shared_ptr<RecorderGraph>(new RecorderGraph(
            //            recorder->GetRecorderGraph()
            //            ));
            // preGraph = shared_ptr<RecorderGraph>(new RecorderGraph(
            //            recorder->GetRecorderGraph()
            //            ));
        }

        //Mat q;
        //recorder->ConvertToStitcher(image->originalExtrinsics, q);
        //originalExtrinsics.push_back(q.clone());

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
    
    auto preview = recorder->GetPreviewImage();
    imwrite("dbg/preview.jpg", preview->image.data);
    
    recorder->Finish();
#ifdef USE_MOTOR
    // Motor
    ConvertToStereo convertToStereo(outStore, leftStore, rightStore);
    Stitcher leftStitcher(leftStore);
    Stitcher rightStitcher(rightStore);
    convertToStereo.Finish();

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
    //cv::ocl::setUseOpenCL(false);
    RegisterCrashHandler();

    //DummyCheckpointStore leftStore;
    //DummyCheckpointStore rightStore;
    //DummyCheckpointStore commonStore;
    
  //  StorageSink storeSink(leftStore, rightStore);
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

   // if(!postProcStore.HasUnstitchedRecording()) {
        cout << "Recording." << endl;
        Record(files);
   // } else {
   //     cout << "Skipping Recording." << endl;
   // }
/*

    if(!leftStore.HasUnstitchedRecording()) {
        cout << "Aligning." << endl;
        GlobalAlignment globalAlignment = GlobalAlignment(postProcStore, leftStore, rightStore);
        globalAlignment.Finish();
    } else {
        cout << "Skipping Aligning." << endl;
    }

    optonaut::ExposureCompensator dummyCompensator;
    MultiRingStitcher leftStitcher(leftStore);
    MultiRingStitcher rightStitcher(rightStore);

    vector<vector<InputImageP>> leftRings;
   	vector<vector<InputImageP>> rightRings;
    map<size_t, double> gains;
    leftStore.LoadStitcherInput(leftRings, gains);
    rightStore.LoadStitcherInput(rightRings, gains);

   	leftStitcher.InitializeForStitching(leftRings, dummyCompensator);
		rightStitcher.InitializeForStitching(rightRings, dummyCompensator);



    auto resLeft = leftStitcher.Stitch(ProgressCallback::Empty);
    auto resRight = rightStitcher.Stitch(ProgressCallback::Empty);
    
    if(!useStitcherSink) {

        if(!leftStore.HasUnstitchedRecording()) {
            cout << "No results." << endl;
            return 0;
        }
        ProgressCallback progress([](float) -> bool {
                    //cout << (int)(progress * 100) << "% ";
                    return true;
                });
        ProgressCallbackAccumulator callbacks(progress, {0.5, 0.5});

        {
            cout << "Start left stitcher." << endl;
            optonaut::Stitcher leftStitcher(leftStore);
            auto left = leftStitcher.Finish(callbacks.At(0), "dbg/left");
            */
/*
            DrawPointsOnPanorama(left->image.data, 
                    ExtractExtrinsics(fun::flat(graph->GetRings())), 
                    intrinsics, imageSize, 1200, left->corner);
            
            DrawPointsOnPanorama(left->image.data, 
                    ExtractExtrinsics(fun::flat(preGraph->GetRings())), 
                    intrinsics, imageSize, 1200, left->corner + Point(0, 25));
            
            DrawPointsOnPanorama(left->image.data, 
                    ExtractExtrinsics(allImages),
                    intrinsics, imageSize, 1200, left->corner + Point(0, -25),
                    Scalar(0xFF, 0x00, 0x00));
            
            DrawPointsOnPanorama(left->image.data, originalExtrinsics,
                    intrinsics, imageSize, 1200, left->corner + Point(0, -100),
                    Scalar(0xFF, 0x00, 0xFF));
*/
        /*
            imwrite("dbg/left.jpg", left->image.data);
            left->image.Unload();  
            left->mask.Unload();  
        }
        {
            cout << "Start right stitcher." << endl;
            optonaut::Stitcher rightStitcher(rightStore);
            auto right = rightStitcher.Finish(callbacks.At(1), "dbg/right");
            imwrite("dbg/right.jpg", right->image.data);    
            right->image.Unload();  
            right->mask.Unload();  
        } 
    } else {
        auto left = stitcherSink.GetLeftResult(); 
        imwrite("dbg/left.jpg", left->image.data);
        auto right = stitcherSink.GetRightResult(); 
        imwrite("dbg/right.jpg", right->image.data);    
    }

    //leftStore.Clear();
    //rightStore.Clear();
    */
    return 0;
}
