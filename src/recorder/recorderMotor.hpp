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
#include "ringCloser.hpp"
#include "jumpFilter.hpp"

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
        FeedbackImageSelector preController;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;
        uint32_t keyframeCount;
        
        AsyncQueue<SelectionInfo> stereoConversionQueue;
        RingProcessor<SelectionInfo> stereoRingBuffer;
        AsyncQueue<StereoImage> saveQueue;
        AsyncQueue<InputImageP> debugQueue;


        std::shared_ptr<AsyncTolerantRingRecorder> previewRecorder;
        RecorderGraph previewGraph;
        
        STimer monoTimer;
        STimer pipeTimer;
        
        InputImageP last;
        int lastRingId;
        
        string debugPath;
        
        
    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;
        
        int uselessVariable = 0;

        static string tempDirectory;
        static string version;

        static bool exposureEnabled;

        Recorder(Mat base, Mat zeroWithoutBase, Mat intrinsics, 
                StereoSink &sink, string debugPath = "",
                int graphConfiguration = RecorderGraph::ModeAll, bool unused = false) :
            base(base),
            sink(sink),
            stereoConverter(),
            isIdle(false),
            isFinished(false),
            hasStarted(false),
            generator(),
            preRecorderGraph(generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityHalf, 0, 8)),
            preController(preRecorderGraph, [this] (const SelectionInfo &x) {
                ForwardToStereoConversionQueue(x);
            }, Vec3d(M_PI / 64, M_PI / 128, M_PI / 16)),
            imagesToRecord(preRecorderGraph.Size()),
            recordedImages(0),
            keyframeCount(0),
            stereoConversionQueue(
                std::bind(&Recorder::ForwardToStereoRingBuffer,
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

            debugQueue([this] (const InputImageP &x) {
                static int debugCounter = 0;
                InputImageToFile(x, 
                        this->debugPath + "/" + ToString(debugCounter++) + ".jpg");
 	    }),
            previewGraph(RecorderGraphGenerator::Sparse(
                        preRecorderGraph, 
                        1,
                        preRecorderGraph.ringCount / 2)),
            lastRingId(-1),
            debugPath(debugPath)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;
            
            // Allocate some useless memory.
            // We do so to "reserve" pages, so we don't have lag when allocating during the first ring.
            {
                vector<void*> uselessMem;
                
                // rings = vector<vector<SelectionPoint>> targets
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
            

            //cout << "Initializing Optonaut Pipe." << endl;
            
            //cout << "Base: " << base << endl;
            //cout << "BaseInv: " << baseInv << endl;
            //cout << "Zero: " << zero << endl;
        
       }
        
        void ForwardToStereoConversionQueue(SelectionInfo in) {
            
            recordedImages++;
						if (in.closestPoint.ringId == preRecorderGraph.GetRings().size()/2) {
                PushToPreview(in);
						}
            if(preRecorderGraph.HasChildRing(in.closestPoint.ringId)) {
                if(keyframeCount % 2 == 0) {
                    //Save some memory by sparse keyframing.
                    if(!in.image->image.IsLoaded()) {
                        in.image->image.Load();
                    }
                }

                keyframeCount++;
            }

            int size = stereoConversionQueue.Push(in);
            if(size > 100) {
                cout << "Warning: Input Queue overflow: " <<  size << endl;
                this_thread::sleep_for(chrono::seconds(1));
            }
        }

        
        RecorderGraph& GetPreRecorderGraph() {
            return preRecorderGraph;
        }
 
        RecorderGraph& GetRecorderGraph() {
            return preRecorderGraph;
        }


        void ForwardToStereoProcess(const SelectionInfo &a, const SelectionInfo &b) {

            SelectionEdge dummy;
            //cout << "Stereo Process received: " << a.closestPoint.globalId << " <> " << b.closestPoint.globalId << endl;
            if(!preRecorderGraph.GetEdge(a.closestPoint, b.closestPoint, dummy)) {
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
            //cout << "returning " << converted.size() << " rings " << endl;
            return converted;
        }

        void Dispose() {
            //cout << "Pipeline Dispose called by " << std::this_thread::get_id() << endl;
            stereoConversionQueue.Dispose();
            saveQueue.Dispose();
        }
        
        void FinishImage(const SelectionInfo &fin) {

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
            
            
            Assert(pair.a.image->IsLoaded());
            Assert(pair.b.image->IsLoaded());

            
            if(exposureEnabled)
                exposure.Register(pair.a.image, pair.b.image);
            
            monoTimer.Tick("Exposure Comp");
            
            StereoImage stereo;
            stereoConverter.CreateStereo(pair.a, pair.b, stereo);

            Assert(stereo.valid);
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
       
        void PushToPreview(SelectionInfo in) {
            if(previewRecorder == nullptr) {
                AutoLoad q(in.image);
                previewRecorder = 
                    std::make_shared<AsyncTolerantRingRecorder>(in, previewGraph);
            }

            previewRecorder->Push(in.image);
        }

        bool PreviewAvailable() {
            return true;
        }

        StitchingResultP FinishPreview() {

            STimer finishPreview;
            
            if(preRecorderGraph.ringCount == 1) {
                // If the input buffer queue is still running,
                // end it. Otherwise, the preview generation already stopped this.
                preController.Flush();
            } 
            stereoConversionQueue.Finish();
            
            Assert(previewRecorder != nullptr);
            
            StitchingResultP res = previewRecorder->Finalize();
            previewRecorder = nullptr;
            finishPreview.Tick("Finish Preview");

            return res;
        }
        
        void ForwardToStereoRingBuffer(const SelectionInfo in) {
            if(lastRingId != (int)in.closestPoint.ringId) {
                stereoRingBuffer.Flush();
                lastRingId = in.closestPoint.ringId;
            }

            stereoRingBuffer.Push(in);
        }
   
        void Push(InputImageP image) {
            //STimer processingTime(true);

            //cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;
            
            if(debugPath != "" && !isIdle) {
                AssertFalseInProduction(false);
                static int debugCounter = 0;
                if (debugCounter % 10 == 0 ) {
                    image->LoadFromDataRef();
                    debugQueue.Push(image);
                }
                debugCounter++;
            }
            
            AssertM(!isFinished, "Warning: Push after finish - this is probably a racing condition");
            
            //pipeTimer.Tick("Push");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
       
            image->originalExtrinsics.copyTo(image->adjustedExtrinsics);
            
            last = image;
            
            Mat rvec;
            ExtractRotationVector(image->originalExtrinsics, rvec);
            
            static STimer frame(true);
            
            //frame.Tick("## Frame Received");
            //cout << "## Rotation X: " << rvec.at<double>(0) << endl;
            //cout << "## Rotation Y: " << rvec.at<double>(1) << endl;
            //cout << "## Rotation Z: " << rvec.at<double>(2) << endl;
            //cout << "## Idle: " << (isIdle ? 1 : 0) << endl;
            
            Assert(image != NULL);
            
            bool shouldLoad = preController.Push(image, isIdle);
            
            if(shouldLoad) {
                //static STimer loadingTime(true);
                if (!image->IsLoaded()) {
                    image->LoadFromDataRef();
                }
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

            // If the input buffer queue is still running,
            // end it. Otherwise, the preview generation already stopped this.
            preController.Flush();
            stereoConversionQueue.Finish();
            
            stereoRingBuffer.Flush();
            saveQueue.Finish();
            debugQueue.Finish();
            
            if(debugPath != "") {
                std::abort();
            }
            
            if(HasResults()) {

                if(exposureEnabled)
                    exposure.FindGains();

                vector<vector<InputImageP>> rightRings = 
                    preRecorderGraph.SplitIntoRings(rightImages);
                vector<vector<InputImageP>> leftRings = 
                    preRecorderGraph.SplitIntoRings(leftImages);

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
