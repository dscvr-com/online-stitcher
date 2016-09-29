#include "../io/inputImage.hpp"
#include "../io/io.hpp"
#include "../io/checkpointStore.hpp"
#include "../stereo/monoStitcher.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/functional.hpp"
#include "../common/queueProcessor.hpp"
#include "../common/asyncQueueWorker.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
#include "../common/reduceProcessor.hpp"
#include "../stitcher/simpleSphereStitcher.hpp"
#include "../stitcher/ringStitcher.hpp"
#include "../debug/debugHook.hpp"
#include "../recorder/stereoSink.hpp"
#include "../recorder/imageSink.hpp"
#include "../common/logger.hpp"

#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "asyncTolerantRingRecorder.hpp"
#include "ringCloser.hpp"
#include "visualStabilizer.hpp"

#include <chrono>

#ifndef OPTONAUT_RECORDER_HEADER
#define OPTONAUT_RECORDER_HEADER

namespace optonaut {
    
    struct StereoPair {
        SelectionInfo a;
        SelectionInfo b;
    };
    
    class Recorder {

    private: 

        Mat base;
        Mat baseInv;
        Mat zero;

        VisualStabilizer stabilizer;

        ImageSink &sink;
        
        vector<InputImageP> leftImages;
        vector<InputImageP> rightImages;
        vector<InputImageP> postImages;

        ExposureCompensator exposure;

        MonoStitcher stereoConverter;

        bool isIdle;
        bool isFinished;
        bool hasStarted;
        
        RecorderGraphGenerator generator;
        RecorderGraph preRecorderGraph;
        FeedbackImageSelector preController;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;
        
        // AsyncQueue<ImageSelector> FeedbackImageSelector;
        
        //AsyncQueue<SelectionInfo> ForwardToPostProcessImageQueue;
        AsyncQueue<InputImageP> debugQueue;
        AsyncQueue<SelectionInfo> postProcessImageQueue;
        AsyncQueue<SelectionInfo> previewImageQueue;

        vector<SelectionInfo> firstRing;
        bool firstRingFinished;
        

        std::shared_ptr<AsyncTolerantRingRecorder> previewRecorder;
        RecorderGraph previewGraph;
        
        STimer monoTimer;
        STimer pipeTimer;
        
        InputImageP last;
        int lastRingId;
        
        string debugPath;

        Mat refinedIntrinsics;
        
    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;
        static Mat androidZero;
        
        int uselessVariable = 0;

        static string tempDirectory;
        static string version;

        static bool exposureEnabled;
        static bool alignmentEnabled;

        // Delta tolerance: 1 for production, higher value for testing on PC. 
        // Makes the recorder select images that are further off selection points, 
        // because we don't have a chance to "correct" the movement of the phone
        // wehn we're debugging on PC. 
        static constexpr double dt = 1.0;

        Recorder(Mat base, Mat zeroWithoutBase, Mat intrinsics, 
                ImageSink &sink, string debugPath = "",
                int graphConfiguration = RecorderGraph::ModeAll
                ) :
            base(base),
            sink(sink),
            stereoConverter(),
            isIdle(false),
            isFinished(false),
            hasStarted(false),
            generator(),
            preRecorderGraph(generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityNormal, 0, 8)),
            preController(preRecorderGraph, [this] (const SelectionInfo &x) {
                ForwardToPostProcessImageQueue(x);
                },
                          
            Vec3d(M_PI / 64 * dt, M_PI / 128 * dt, M_PI / 16 * dt)),
            imagesToRecord(preRecorderGraph.Size()),
            recordedImages(0),
            debugQueue([this] (const InputImageP &x) {
                static int debugCounter = 0;
                InputImageToFile(x, 
                        this->debugPath + "/" + ToString(debugCounter++) + ".jpg");
 	           }),
            postProcessImageQueue([this] (const SelectionInfo &x) {
                AssertNEQ(x.image, InputImageP(NULL));
                SavePostProcessImage(x);
 	          }),
            previewImageQueue([this] (const SelectionInfo &x) {
                AssertNEQ(x.image, InputImageP(NULL));
                SavePreviewImage(x);
              }),

            firstRingFinished(false),
            previewGraph(RecorderGraphGenerator::Sparse(
                        preRecorderGraph, 
                        1,
                        preRecorderGraph.ringCount / 2)),
            lastRingId(-1),
            debugPath(debugPath),
            refinedIntrinsics(0, 0, CV_64F)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;

            // Make sure we don't use debugger params in production. 
            AssertFalseInProduction(dt != 1.0);
            
            // Allocate some useless memory.
            // We do so to "reserve" pages, so we don't have lag 
            // when allocating during the first ring.
            {
                vector<void*> uselessMem;
                
                auto &rings = preRecorderGraph.GetRings();
                
                // Allocate a little extra for working/preview
                for(size_t i = 0; i < rings[rings.size() / 2].size() + 20; i++) {
                    size_t size = 4 * 1280 * 720;
                    void *memory = malloc(size);
                    if(memory == NULL) {
                        AssertM(memory != NULL, "Failed to pre-allocate memory");
                    }
                    memset(memory, 'o', size);
                    uselessVariable += ((int*)memory)[2];
                    uselessMem.push_back(memory);
                }
                
                for(size_t i = 0; i < uselessMem.size(); i++) {
                    free(uselessMem[i]);
                }
            }
            
            Log << "[Recorder] Initialized by thread " << std::this_thread::get_id();
        }
  
        void ForwardToPostProcessImageQueue(SelectionInfo in) {
                  
            auto image = in.image;

            recordedImages++;
            
            SCounters::Increase("Used Images");

            postProcessImageQueue.Push(in);
            if(!firstRingFinished) {
                previewImageQueue.Push(in);
            } else {
                if(refinedIntrinsics.cols != 0) {
                    refinedIntrinsics.copyTo(image->intrinsics);
                }
            }
       
        }
        
        RecorderGraph& GetPreRecorderGraph() {
            return preRecorderGraph;
        }

        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
        }
        
        void ConvertToStitcher(const Mat &in, Mat &out) const {
            out = (base * zero * in.inv() * baseInv);
        }
      
        Mat GetBallPosition() const {
            return ConvertFromStitcher(preController.GetBallPosition());
        }
        
        SelectionInfo GetCurrentKeyframe() const {
            return preController.GetCurrent();
        }
        
        double GetDistanceToBall() const {
            return preController.GetError();
        }
        
        const Mat &GetAngularDistanceToBall() const {
            return preController.GetErrorVector();
        }

        //Methods already coordinates in input base. 
        Mat GetOrigin() const {
            return baseInv * zero * base;
        }

        Mat GetCurrentRotation() const {
            return ConvertFromStitcher(last->originalExtrinsics);
        }

        vector<SelectionPoint> GetSelectionPoints() const {
            vector<SelectionPoint> converted;
            for(auto ring : preRecorderGraph.GetRings()) {
                ring.push_back(ring.front());
                for(auto point : ring) {
                    SelectionPoint n;
                    n.globalId = point.globalId;
                    n.ringId = point.ringId;
                    n.localId = point.localId;
                    n.extrinsics = ConvertFromStitcher(point.extrinsics);
                    
                    converted.push_back(n);
                }
                
            }
            return converted;
        }

        void Dispose() {
            Log << "[Recorder] Dispose by thread " << std::this_thread::get_id();
            
            debugQueue.Dispose();
            postProcessImageQueue.Dispose();
        }
        
        void SavePostProcessImage(SelectionInfo in) {
            sink.Push(in);
            auto copy = std::make_shared<InputImage>(*in.image);
            copy->image = Image();
            postImages.push_back(copy);
            
        }
 
        void SavePreviewImage(SelectionInfo in) {
            if(firstRing.size() == 0 || 
                firstRing.back().closestPoint.ringId == in.closestPoint.ringId) {
                firstRing.push_back(in);
                // moved the push to preview here to minimize the 
                // lag when finishing the first ring
                PushToPreview(in);
            } else {
                FinishFirstRing(); 
            }
        }
        
        void PushToPreview(SelectionInfo in) {
            if(previewRecorder == nullptr) {
                AutoLoad q(in.image);
                previewRecorder = 
                    std::make_shared<AsyncTolerantRingRecorder>(in, previewGraph);
            }

            previewRecorder->Push(in.image);
        }

        bool PreviewAvailable() {
   
            return previewRecorder != nullptr;
        }

        StitchingResultP FinishPreview() {

            STimer finishPreview;
            
            if(preRecorderGraph.ringCount == 1) {
                // If the input buffer queue is still running,
                // end it. Otherwise, the preview generation already stopped this.
                preController.Flush();
                if(!firstRingFinished) {
                    FinishFirstRing();
                }
            } 
            
            Assert(previewRecorder != nullptr);
            
            StitchingResultP res = previewRecorder->Finalize();
            previewRecorder = nullptr;
            finishPreview.Tick("Finish Preview");

            return res;
        }
        
        void ForwardToStereoRingBuffer(const SelectionInfo in) {
            if(lastRingId != (int)in.closestPoint.ringId) {
                lastRingId = in.closestPoint.ringId;
            }
        }

        //Async
        void FinishFirstRing() {
            firstRingFinished = true; 

            Log << "[Recorder] First ring finishing by thread " << 
                std::this_thread::get_id();

            if (firstRing.size() == 0 ) {
               firstRing.clear();
               return;
           }


            RingCloser::CloseRing(fun::map<SelectionInfo, InputImageP>(firstRing, 
            [](const SelectionInfo &x) {
                return x.image;
            }));
            refinedIntrinsics = Mat::eye(3, 3, CV_64F);
            firstRing[0].image->intrinsics.copyTo(refinedIntrinsics);
            
            firstRing.clear();
            
            Log << "[Recorder] First ring finished by thread " << 
                std::this_thread::get_id();
        }
    
        void Push(InputImageP image) {
            Log << "[Recorder] Got image by thread " << 
                std::this_thread::get_id();

            Mat extrinsics = image->originalExtrinsics;

            image->originalExtrinsics = Mat(4, 4, CV_64F);
            image->adjustedExtrinsics = Mat(4, 4, CV_64F);

            extrinsics.copyTo(image->originalExtrinsics);
            extrinsics.copyTo(image->adjustedExtrinsics);

            if(debugPath != "" && !isIdle) {

                AssertFalseInProduction(true);

                image->LoadFromDataRef();
                // create a copy of the image
                InputImageP copy(new InputImage());
                copy->image = Image(image->image);
                copy->dataRef = image->dataRef;
                copy->originalExtrinsics = image->originalExtrinsics;
                copy->adjustedExtrinsics = image->adjustedExtrinsics;
                copy->intrinsics = image->intrinsics.clone();
                copy->exposureInfo = image->exposureInfo;
                copy->id = image->id;
                debugQueue.Push(copy);
            }

            AssertM(!isFinished, "Warning: Push after finish - this is probably a racing condition");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
            
            if (!image->IsLoaded()) {
                //static STimer loadingTime(true);
                image->LoadFromDataRef();
                //loadingTime.Tick("## Loading Time");
            }
            
            if(hasStarted)
            {
                //stabilizer.Push(image);
                //stabilizer.GetCurrentEstimate().copyTo(image->originalExtrinsics);
            }
            image->originalExtrinsics.copyTo(image->adjustedExtrinsics);

            last = image;
            
            //Mat rvec;
            //ExtractRotationVector(image->originalExtrinsics, rvec);
            
            //static STimer frame(true);
            
            //frame.Tick("## Frame Received");
            //cout << "## Rotation X: " << rvec.at<double>(0) << endl;
            //cout << "## Rotation Y: " << rvec.at<double>(1) << endl;
            //cout << "## Rotation Z: " << rvec.at<double>(2) << endl;
            //cout << "## Idle: " << (isIdle ? 1 : 0) << endl;
            
            Assert(image != NULL);
            
            if(preController.Push(image, isIdle)) {
                hasStarted = true;
            }
            
            if(preController.IsFinished()) {
                isFinished = true;
            }
            
            //processingTime.Tick("## Processing Time");
        }
        
        void Cancel() {
            Log << "[Recorder] Pipeline cancel called by " << std::this_thread::get_id();

//        void Cancel() {
//            cout << "Pipeline Cancel called by " << std::this_thread::get_id() << endl;
            isFinished = true;

            debugQueue.Finish();
            postProcessImageQueue.Finish();
            previewImageQueue.Finish();
            
            if(debugPath != "") {
                std::abort();
            }
        }
        
        void Finish() {
            Log << "[Recorder] Pipeline Finish called by " << std::this_thread::get_id();
            isFinished = true;

            if(!firstRingFinished) {
                FinishFirstRing();
            }
            
            debugQueue.Finish();
            postProcessImageQueue.Finish();
            previewImageQueue.Finish();
            
            if(debugPath != "") {
                std::abort();
            }
            
            if(HasResults()) {
                if(exposureEnabled)
                    exposure.FindGains();

                vector<vector<InputImageP>> postRings = 
                    preRecorderGraph.SplitIntoRings(postImages);

                sink.Finish(postRings, exposure.GetGains());
            } else {
                AssertWM(false, "No results in recorder.");
            }
        }
        ExposureInfo GetExposureHint() {
            Assert(false); //Wrong semantics.
            const Mat current; //= aligner->GetCurrentBias();
            
            vector<KeyframeInfo> frames; // = aligner->GetClosestKeyframes(current, 2);
            
            if(frames.size() < 2) {
                return ExposureInfo();
            } else {
                ExposureInfo info;
                auto af = frames[0].keyframe;
                auto bf = frames[1].keyframe;
                auto a = af->exposureInfo;
                auto b = bf->exposureInfo;
                
                auto atos = abs(GetDistanceX(current, af->adjustedExtrinsics));
                auto btos = abs(GetDistanceX(current, bf->adjustedExtrinsics));
                
                auto fa = atos / (atos + btos);
                auto fb = btos / (atos + btos);
                
                info.exposureTime = (a.exposureTime * fa + b.exposureTime * fb);
                info.iso = (a.iso * fa + b.iso * fb);
                info.gains.red = (a.gains.red * fa  + b.gains.red * fb);
                info.gains.blue  = (a.gains.blue * fa + b.gains.blue * fb);
                info.gains.green  = (a.gains.green * fa + b.gains.green * fb);
                
                cout << "ios: " << info.iso << endl;
                
                return info;
            }
        }
        
        bool HasResults() {
            return postImages.size() > 0;
        }
                
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            SelectionEdge dummy; 
            return preRecorderGraph.GetEdge(a, b, dummy);
        }
        
        bool IsIdle() {
            return isIdle;
        }
        
        bool HasStarted() {
            return hasStarted;
        }

        bool IsFinished() {
            return isFinished;
        }
        
        void SetIdle(bool isIdle) {
            this->isIdle = isIdle;
        }
        
        uint32_t GetImagesToRecordCount() {
            return imagesToRecord;
        }
        
        uint32_t GetRecordedImagesCount() {
            return recordedImages;
        }
    };    
}

#endif
