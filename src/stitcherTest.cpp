#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "recorder/recorder.hpp"
#include "recorder/storageSink.hpp"
#include "stitcher/stitcherSink.hpp"
#include "common/intrinsics.hpp"
#include "common/backtrace.hpp"
#include "common/drawing.hpp"
#include "stitcher/stitcher.hpp"
#include "io/io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

shared_ptr<RecorderGraph> graph;
shared_ptr<RecorderGraph> preGraph;
Mat intrinsics;
Size imageSize;
vector<InputImageP> allImages;
vector<Mat> originalExtrinsics;

void Record(vector<string> &files, StereoSink &sink) {

    if(files.size() == 0) {
        cout << "No Input." << endl;
        return;
    }

    static const bool isAsync = true;
    shared_ptr<Recorder> recorder(NULL);

    //DebugHook::Instance = &hook;

    for(size_t i = 0; i < files.size(); i++) {
        auto lt = system_clock::now();
        auto image = InputImageFromFile(files[i], false);
        cout << "[Record] InputImageFromFile :" << files[i] << endl;
            
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
            recorder = shared_ptr<Recorder>(
                    new Recorder(Recorder::iosBase, Recorder::androidZero, 
                        image->intrinsics, sink, "", RecorderGraph::ModeCenter, 
                        isAsync));

            // Needed for debug. 
            intrinsics = image->intrinsics;
            imageSize = Size(intrinsics.at<double>(0, 2), 
                    intrinsics.at<double>(1, 2));
            graph = shared_ptr<RecorderGraph>(new RecorderGraph(
                        recorder->GetRecorderGraph()
                        ));
            preGraph = shared_ptr<RecorderGraph>(new RecorderGraph(
                        recorder->GetPreRecorderGraph()
                        ));
        }

        Mat q;
        recorder->ConvertToStitcher(image->originalExtrinsics, q);
        originalExtrinsics.push_back(q.clone());

        recorder->Push(image);

        tmpMat.release();
        
        if(recorder->IsFinished()) {
            break;
        }

        if(isAsync) {
            auto now = system_clock::now(); 
            auto diff = now - lt;
            auto sleep = 10ms - diff;
            //cout << "Sleeping for " << duration_cast<microseconds>(sleep).count() << endl;
            this_thread::sleep_for(sleep);
        }
    }

    if(recorder->PreviewAvailable()) {
        auto preview = recorder->FinishPreview();
        imwrite("dbg/preview.jpg", preview->image.data);
    }

    recorder->Finish();

    //hook.Draw();
    //hook.WaitForExit();

    recorder->Dispose();
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    RegisterCrashHandler();

    static const bool useStitcherSink = false;

    CheckpointStore leftStore("tmp/left/", "tmp/shared/");
    CheckpointStore rightStore("tmp/right/", "tmp/shared/");
    CheckpointStore commonStore("tmp/common/", "tmp/shared/");
    
    //DummyCheckpointStore leftStore;
    //DummyCheckpointStore rightStore;
    //DummyCheckpointStore commonStore;
    
    StorageSink storeSink(leftStore, rightStore);
    StitcherSink stitcherSink;

    cout << "Starting." << endl;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        cout << "imageName :" << imageName << endl;
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);

    if(!leftStore.HasUnstitchedRecording()) {
        cout << "Recording." << endl;
        if(useStitcherSink) {
            Record(files, stitcherSink);
        } else {
            Record(files, storeSink);
        }
    }
    
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
    
    return 0;
}
