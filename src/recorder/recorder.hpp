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

#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "streamingRecorderController.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "asyncTolerantRingRecorder.hpp"
#include "iterativeBundleAligner.hpp"

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

        shared_ptr<Aligner> aligner;

        StereoSink &sink;
        
        vector<InputImageP> leftImages;
        vector<InputImageP> rightImages;

        ExposureCompensator exposure;

        MonoStitcher stereoConverter;

        bool isIdle;
        bool isFinished;
        bool hasStarted;
        
        RecorderGraphGenerator generator;
        RecorderGraph preRecorderGraph;
        RecorderGraph recorderGraph;
        FeedbackImageSelector preController;
        ImageSelector recorderController;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;
        uint32_t keyframeCount;
        
        AsyncQueue<SelectionInfo> inputBufferQueue;
        QueueProcessor<SelectionInfo> alignerDelayQueue;
        AsyncQueue<SelectionInfo> stereoConversionQueue;
        RingProcessor<SelectionInfo> stereoRingBuffer;
        AsyncQueue<StereoImage> saveQueue;

        vector<SelectionInfo> firstRing;
        vector<InputImageP> firstRingImagePool;

        std::shared_ptr<AsyncTolerantRingRecorder> previewRecorder;
        RecorderGraph previewGraph;
        bool previewImageAvailable;
        
        STimer monoTimer;
        STimer pipeTimer;
        
        InputImageP last;
        
        string debugPath;
        
        InputImageP GetParentKeyframe(const Mat &extrinsics) {
            return aligner->GetClosestKeyframe(extrinsics);
        }
        
    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;

        static string tempDirectory;
        static string version;

        static bool exposureEnabled;
        static bool alignmentEnabled;
        
        Recorder(Mat base, Mat zeroWithoutBase, Mat intrinsics, 
                StereoSink &sink, string debugPath = "",
                int graphConfiguration = RecorderGraph::ModeAll, 
                bool isAsync = true) :
            base(base),
            sink(sink),
            stereoConverter(),
            isIdle(false),
            isFinished(false),
            hasStarted(false),
            generator(),
            preRecorderGraph(generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityDouble, 0, 6)),
            recorderGraph(RecorderGraphGenerator::Sparse(preRecorderGraph, 3)),
            preController(preRecorderGraph, [this] (const SelectionInfo &x) {
                inputBufferQueue.Push(x);
            }, Vec3d(M_PI / 64, M_PI / 128, M_PI / 16)),
            recorderController(recorderGraph, [this] (const SelectionInfo &x) {
                ForwardToStereoConversionQueue(x);
            }, Vec3d(M_PI / 16, M_PI / 16, M_PI / 8), false),
            imagesToRecord(preRecorderGraph.Size()),
            recordedImages(0),
            keyframeCount(0),
            inputBufferQueue([this] (const SelectionInfo &x) {
                ForwardToAligner(x);
            }),
            alignerDelayQueue(15,
                std::bind(&Recorder::ApplyAlignment,
                          this, placeholders::_1)),
            stereoConversionQueue(
                std::bind(&Recorder::ForwardToMonoQueue,
                    this, placeholders::_1)),
            stereoRingBuffer(1,
                std::bind(&Recorder::ForwardToStereoProcess,
                    this, placeholders::_1,
                    placeholders::_2),
                std::bind(&Recorder::FinishImage, 
                    this, placeholders::_1)),
            saveQueue(
                    std::bind(&Recorder::SaveStereoResult,
                    this, placeholders::_1)),
            previewGraph(RecorderGraphGenerator::Sparse(
                        recorderGraph, 
                        1,
                        recorderGraph.ringCount / 2)),
            previewImageAvailable(false),
            debugPath(debugPath)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;

            //cout << "Initializing Optonaut Pipe." << endl;
            
            //cout << "Base: " << base << endl;
            //cout << "BaseInv: " << baseInv << endl;
            //cout << "Zero: " << zero << endl;
        
            if(Recorder::alignmentEnabled) {
                aligner = shared_ptr<Aligner>(new RingwiseStreamAligner(recorderGraph, exposure, isAsync));
            } else {
                aligner = shared_ptr<Aligner>(new TrivialAligner());
            }
        }
        
        void ForwardToStereoConversionQueue(SelectionInfo in) {
            int size = stereoConversionQueue.Push(in);
            if(size > 100) {
                cout << "Warning: Input Queue overflow: " <<  size << endl;
                this_thread::sleep_for(chrono::seconds(1));
            }
        }

        void ForwardToAligner(SelectionInfo in) {
            recordedImages++;
            SCounters::Increase("Used Images");
            aligner->Push(in.image);
            alignerDelayQueue.Push(in);
        }

        RecorderGraph& GetRecorderGraph() {
            return recorderGraph;
        }
        
        RecorderGraph& GetPreRecorderGraph() {
            return preRecorderGraph;
        }

        void ForwardToStereoProcess(const SelectionInfo &a, const SelectionInfo &b) {

            SelectionEdge dummy;
            //cout << "Stereo Process received: " << a.closestPoint.globalId << " <> " << b.closestPoint.globalId << endl;
            if(!recorderGraph.GetEdge(a.closestPoint, b.closestPoint, dummy)) {
                cout << "Warning: Unordered." << endl;
                return;
            }

            StereoPair pair;
            //cout << "Pairing: " << a.closestPoint.globalId << " <> " << b.closestPoint.globalId << endl;
            AssertNEQ(a.image, InputImageP(NULL));
            AssertNEQ(b.image, InputImageP(NULL));
            pair.a = a;
            pair.b = b;
        
            ConvertToStereo(pair);
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
            return ConvertFromStitcher(last->adjustedExtrinsics);
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
            //cout << "returning " << converted.size() << " rings " << endl;
            return converted;
        }

        void Dispose() {
            //cout << "Pipeline Dispose called by " << std::this_thread::get_id() << endl;
            inputBufferQueue.Dispose();
            aligner->Dispose();
            stereoConversionQueue.Dispose();
            saveQueue.Dispose();
        }
        
        void FinishImage(const SelectionInfo &fin) {
            ExposureFeed(fin.image);
        }   
        
        void ConvertToStereo(const StereoPair &pair) {
            
            monoTimer.Reset();
            
            if(!pair.a.image->IsLoaded()) {
                pair.a.image->image.Load();
            }
            if(!pair.b.image->IsLoaded()) {
                pair.b.image->image.Load();
            }
            monoTimer.Tick("Load Input");
            
            
            assert(pair.a.image->IsLoaded());
            assert(pair.b.image->IsLoaded());

            
            if(exposureEnabled)
                exposure.Register(pair.a.image, pair.b.image);
            
            monoTimer.Tick("Exposure Comp");
            
            StereoImage stereo;
            stereoConverter.CreateStereo(pair.a, pair.b, stereo);

            assert(stereo.valid);
            monoTimer.Tick("Stereo Conv");
            
            int size = saveQueue.Push(stereo);
            
            if(size > 20) {
                cout << "Warning: Output Queue overflow: " <<  size << endl;
            }
        }
        
        void SaveStereoResult(StereoImage stereo) {
            sink.Push(stereo);
            
            leftImages.push_back(stereo.A);
            rightImages.push_back(stereo.B);
        }
        
        void ExposureFeed(InputImageP a) {
            
            if(!exposureEnabled)
                return;
            
            InputImageP b = GetParentKeyframe(a->adjustedExtrinsics);
            if(b != NULL && a->id != b->id) {
                exposure.Register(a, b);
            }
        }

        void PushToPreview(SelectionInfo in) {
            if(previewRecorder == nullptr) {
                AutoLoad q(in.image);
                previewRecorder = 
                    std::make_shared<AsyncTolerantRingRecorder>(in, previewGraph);
                
                previewImageAvailable = true;
            }

            previewRecorder->Push(in.image);
        }

        bool PreviewAvailable() {
            return previewImageAvailable;
        }

        StitchingResultP FinishPreview() {

            STimer finishPreview;
            Assert(previewRecorder != nullptr);
            
            if(recorderGraph.ringCount == 1) {
                // If we only have a single ring, we have to empty this queue
                // before blending the preview to avoid racing conditions.
                inputBufferQueue.Finish();
                aligner->Finish();
                alignerDelayQueue.Flush();
                stereoConversionQueue.Finish();
            }
            
            StitchingResultP res = previewRecorder->Finalize();
            previewRecorder = nullptr;
            previewImageAvailable = false;
            finishPreview.Tick("Finish Preview");

            return res;
        }
        
        void ForwardToMonoQueue(const SelectionInfo in) {
            if(recorderGraph.HasChildRing(in.closestPoint.ringId)) {
                if(keyframeCount % 2 == 0) {
                    //Save some memory by sparse keyframing.
                    if(!in.image->image.IsLoaded()) {
                        in.image->image.Load();
                    }
                    aligner->AddKeyframe(CloneAndDownsample(in.image));
                }

                keyframeCount++;
            }

            stereoRingBuffer.Push(in);
        }

        void ApplyAlignment(SelectionInfo info) {
            
            auto image = info.image;

            image->adjustedExtrinsics = 
                aligner->GetCurrentBias() * image->originalExtrinsics;

            recorderController.Push(image);

            // TODO - Necassary?
            // if(isIdle)
            //    return;

            if(DebugHook::Instance != NULL) {
                DebugHook::Instance->RegisterImageRotationModel(
                        image->image.data, 
                        ConvertFromStitcher(image->adjustedExtrinsics), 
                        image->intrinsics);
            }
        }
    
        void Push(InputImageP image) {
            //STimer processingTime(true);

            //cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;
            
            last = image;
            
            if(debugPath != "" && !isIdle) {
                AssertFalseInProduction(false);
                static int debugCounter = 0;
                image->LoadFromDataRef();
                InputImageToFile(image, 
                        debugPath + "/" + ToString(debugCounter++) + ".jpg");
            }
            
            AssertM(!isFinished, "Warning: Push after finish - this is probably a racing condition");
            
            //pipeTimer.Tick("Push");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
            image->adjustedExtrinsics = image->originalExtrinsics;
            
            
            Mat rvec;
            ExtractRotationVector(image->originalExtrinsics, rvec);
            
            static STimer frame(true);
            
            //frame.Tick("## Frame Received");
            //cout << "## Rotation X: " << rvec.at<double>(0) << endl;
            //cout << "## Rotation Y: " << rvec.at<double>(1) << endl;
            //cout << "## Rotation Z: " << rvec.at<double>(2) << endl;
            //cout << "## Idle: " << (isIdle ? 1 : 0) << endl;
            
            bool shouldLoad = preController.Push(image, isIdle);
            
            if(shouldLoad) {
                //static STimer loadingTime(true);
                image->LoadFromDataRef();
                //SCounters::Increase("Loaded Images");
                hasStarted = true;
                //loadingTime.Tick("## Loading Time");
            }
            
            if(preController.IsFinished()) {
                isFinished = true;
            }
            
            //processingTime.Tick("## Processing Time");
        }

        void Finish() {
            //cout << "Pipeline Finish called by " << std::this_thread::get_id() << endl;
            isFinished = true;

            if(stereoConversionQueue.IsRunning()) {
                // If the stereo conversion queue is still running,
                // end it. Otherwise, the preview generation already stopped this.
                inputBufferQueue.Finish();
                aligner->Finish();
                alignerDelayQueue.Flush();
                stereoConversionQueue.Finish();
            }
            //if(!firstRingFinished) {
            //    FinishFirstRing();
            //}
            
            recorderController.Flush();
            stereoRingBuffer.Flush();
            saveQueue.Finish();
            
            if(debugPath != "") {
                std::abort();
            }
            
            if(HasResults()) {

                if(exposureEnabled)
                    exposure.FindGains();
               
                aligner->Postprocess(leftImages);
                aligner->Postprocess(rightImages);

                vector<vector<InputImageP>> rightRings = 
                    recorderGraph.SplitIntoRings(rightImages);
                vector<vector<InputImageP>> leftRings = 
                    recorderGraph.SplitIntoRings(leftImages);

                sink.Finish(leftRings, rightRings, exposure.GetGains());
            } else {
                AssertWM(false, "No results in recorder.");
            }
        }

        bool HasResults() {
            return leftImages.size() > 0;
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
        
        ExposureInfo GetExposureHint() {
            assert(false); //Wrong semantics. 
            const Mat &current = aligner->GetCurrentBias();
            
            vector<KeyframeInfo> frames = aligner->GetClosestKeyframes(current, 2);
            
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
