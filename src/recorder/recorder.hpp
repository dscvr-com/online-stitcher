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
#include "../debug/debugHook.hpp"

#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "streamingRecorderController.hpp"
#include "tolerantRecorderController.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"

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
        
        CheckpointStore leftStore;
        CheckpointStore rightStore;
        CheckpointStore commonStore;
        
        vector<InputImageP> leftImages;
        vector<InputImageP> rightImages;

        ExposureCompensator exposure;
        
        MonoStitcher stereoConverter;

        bool isIdle;
        bool isFinished;
        bool firstRingFinished;
        bool hasStarted;
        
        RecorderGraphGenerator generator;
        RecorderGraph recorderGraph;
        TolerantRecorderController controller;
        RecorderGraph preRecorderGraph;
        StreamingRecorderController preController;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;
        uint32_t keyframeCount;
        int lastRingId;
        
        ReduceProcessor<SelectionInfo, SelectionInfo> preSelectorQueue;
        QueueProcessor<InputImageP> alignerDelayQueue;
        ReduceProcessor<SelectionInfo, SelectionInfo> selectorQueue;
        AsyncQueue<SelectionInfo> stereoConversionQueue;
        RingProcessor<SelectionInfo> stereoRingBuffer;
        AsyncQueue<StereoImage> saveQueue;

        vector<SelectionInfo> firstRing;
        vector<InputImageP> firstRingImagePool;
        
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
                CheckpointStore &leftStore, CheckpointStore &rightStore, CheckpointStore &commonStore, string debugPath = "",
                int graphConfiguration = RecorderGraph::ModeAll, 
                bool isAsync = true) :
            base(base),
            leftStore(leftStore),
            rightStore(rightStore),
            commonStore(commonStore),
            stereoConverter(),
            isIdle(false),
            isFinished(false),
            firstRingFinished(false),
            hasStarted(false),
            generator(),
            recorderGraph(generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityNormal)),
            controller(recorderGraph),
            preRecorderGraph(generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityDouble, 8)),
            preController(preRecorderGraph),
            imagesToRecord(preRecorderGraph.Size()),
            recordedImages(0),
            keyframeCount(0),
            lastRingId(-1),
            preSelectorQueue(
                std::bind(&Recorder::SelectBetterMatchForPreSelection,
                    this, placeholders::_1,
                    placeholders::_2), SelectionInfo()),
            alignerDelayQueue(15,
                std::bind(&Recorder::ApplyAlignment,
                          this, placeholders::_1)),
            selectorQueue(
                std::bind(&Recorder::SelectBetterMatchForMonoQueue,
                    this, placeholders::_1,
                          placeholders::_2), SelectionInfo()),
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

        void ForwardToAligner(InputImageP in) {
            aligner->Push(in);
            alignerDelayQueue.Push(in);
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
       
        //TODO: Rename/Remove all ball methods.  
        Mat GetNextKeyframe() const {
            return ConvertFromStitcher(preController.GetNextKeyframe());
        }
        
        double GetDistanceToNextKeyframe() const {
            return preController.GetError();
        }
        
        const Mat &GetAngularDistanceToNextKeyframe() const {
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
            STimer saveTimer;
            leftStore.SaveRectifiedImage(stereo.A);
            rightStore.SaveRectifiedImage(stereo.B);
            
            stereo.A->image.Unload();
            stereo.B->image.Unload();
            
            leftImages.push_back(stereo.A);
            rightImages.push_back(stereo.B);
            monoTimer.Tick("Saved");
        }
        
        void ExposureFeed(InputImageP a) {
            
            if(!exposureEnabled)
                return;
            
            InputImageP b = GetParentKeyframe(a->adjustedExtrinsics);
            if(b != NULL && a->id != b->id) {
                exposure.Register(a, b);
            }
        }
        
        void ForwardToMonoQueue(const SelectionInfo &in) {
            
            if(lastRingId == -1) {
                lastRingId = in.closestPoint.ringId;
            }
            
            if(lastRingId != (int)in.closestPoint.ringId) {
                if(!firstRingFinished) {
                    FinishFirstRing();
                }
                stereoRingBuffer.Flush();
                lastRingId = in.closestPoint.ringId;
            }
            
            if(!firstRingFinished) {
                if(firstRing.size() == 0
                        || firstRing.back().closestPoint.globalId != 
                        in.closestPoint.globalId) {
                    firstRing.push_back(in);    
                }
                firstRingImagePool.push_back(in.image);
                commonStore.SaveStitcherTemporaryImage(in.image->image);
                in.image->image.Unload();
            } else {
                ForwardToMonoQueueEx(in);
            }
        }

        void FinishFirstRing() {
            AssertM(!firstRingFinished, "First ring has not been closed");
            firstRingFinished = true;

            PairwiseCorrelator corr;
            firstRingImagePool.back()->image.Load();
            firstRingImagePool.front()->image.Load();
            auto result = corr.Match(firstRingImagePool.back(), firstRingImagePool.front(), 0, 0, true); 
            size_t n = firstRingImagePool.size();

            if(!result.valid) {
                cout << "Ring closure rejection because: " << result.rejectionReason << endl;
            }

            cout << "Y horizontal angular offset: " << result.angularOffset.x << endl;

            // TODO - SMTH is wrong here. 
            for(size_t i = 0; i < n; i++) {
                double ydiff = -result.angularOffset.x * 
                    (1.0 - ((double)i) / ((double)n));
                //double ydiff = 0;
                Mat correction;
                CreateRotationY(ydiff, correction);
                firstRingImagePool[i]->adjustedExtrinsics = correction * 
                    firstRingImagePool[i]->adjustedExtrinsics;

                cout << "Applying correction of " << ydiff << " to image " << firstRingImagePool[i]->id << endl;
            }

    
            //Re-select images.
           
            size_t m = firstRing.size();
            for(size_t i = 0; i < m; i++) {
                int picked = -1;
                double distCur = 100;
                for(size_t j = 0; j < n; j++) {

                    if(firstRingImagePool[j] == NULL)
                        continue;

                    double distCand = abs(GetAngleOfRotation(
                            firstRingImagePool[j]->adjustedExtrinsics, 
                            firstRing[i].closestPoint.extrinsics));

                    if(distCur > distCand || picked == -1) {
                        distCur = distCand;
                        picked = (int)j;
                    }
                }
                //cout << "Selection Point " << firstRing[i].closestPoint.globalId
                //    << " reassigned image " << firstRing[i].image->id << " to "
                //    << firstRingImagePool[picked]->id << " with dist: "
                //    << distCur << endl;
                AssertGTM(picked, -1, "Must pick image"); 
                firstRing[i].image = firstRingImagePool[picked];
                firstRing[i].dist = distCur;
                firstRingImagePool[picked] = NULL;
            }

            for(size_t i = 0; i < m; i++) {
                ForwardToMonoQueueEx(firstRing[i]);
                firstRing[i].image = NULL; //Decrement our ref to the loaded instance
            }
           
            firstRingImagePool.clear();
            firstRing.clear();
        }

        void ForwardToMonoQueueEx(const SelectionInfo &in) {
           
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

        SelectionInfo SelectBetterMatchForPreSelection(const SelectionInfo &best, const SelectionInfo &in) {

            SCounters::Increase("Better Match Pre Selection In");

            //Init case
            if(!best.isValid)
                return in;
            
            //Invalid case
            if(!in.isValid)
                return best;

            //Can do deffered loading here - we're still on main thread.  
            if(!in.image->IsLoaded()) {
                in.image->LoadFromDataRef();
            }
           
            //Push images that get closer to the target towards the aligner.
            //But only if we're not in debug.
            if(debugPath == "") {
                SCounters::Increase("Better Match Pre Selection Found");
                ForwardToAligner(in.image);
            }
            if(best.closestPoint.globalId != in.closestPoint.globalId) {
                //This delays.
                recordedImages++;
                if(preController.IsFinished()) {
                    isFinished = true;
                }
            }
            
            return in;
        }
        
        SelectionInfo SelectBetterMatchForMonoQueue(const SelectionInfo &best, const SelectionInfo &in) {
                
            SCounters::Increase("Better Match Mono Queue In");
            //Init case
            if(!best.isValid) {
                SCounters::Increase("Better Mono Queue Init");
                return in;
            }

            //Invalid case
            if(!in.isValid) {
                SCounters::Increase("Better Mono Queue Invalid");
                return best;
            }
            
            if(best.closestPoint.globalId != in.closestPoint.globalId) {
                SCounters::Increase("Better Match Mono Queue Found");
                //We will not get a better match. Can forward best.  
                ForwardToStereoConversionQueue(best);
                //ForwardToMonoQueue(in, true);
                //We Finished a ring. Flush.

            } else {
                SCounters::Increase("Better Match Mono Queue Rejected Because Same Id");
                if(!isFinished) {
                    AssertM(best.dist >= in.dist, "Recorder controller only returns better or equal matches");
                }
            }
            //Forward all for testing.  
            //if(!firstRingFinished) {
            //    ForwardToMonoQueue(in);
            //}

            return in; 
        }

        void ApplyAlignment(InputImageP image) {

            image->adjustedExtrinsics = aligner->GetCurrentBias() * image->originalExtrinsics;

            SelectionInfo current = controller.Push(image);

            if(isIdle)
                return;

            if(DebugHook::Instance != NULL) {
                DebugHook::Instance->RegisterImageRotationModel(
                        image->image.data, 
                        ConvertFromStitcher(image->adjustedExtrinsics), 
                        image->intrinsics);
            }
           
            selectorQueue.Push(current); 
        }
    
        void Push(InputImageP image) {
            //cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;
            
            last = image;
            
            if(debugPath != "" && !isIdle) {
                static int debugCounter = 0;
                image->LoadFromDataRef();
                InputImageToFile(image, debugPath + "/" + ToString(debugCounter++) + ".jpg");
            }
            
            AssertM(!isFinished, "Warning: Push after finish - this is probably a racing condition");
            
            //pipeTimer.Tick("Push");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
            image->adjustedExtrinsics = image->originalExtrinsics;
           
            //This preselects.
            if(!preController.IsInitialized() || !hasStarted)
                preController.Initialize(image->adjustedExtrinsics);
            
            //Pass idle info, so we can get UI feedback without modifying state
            SelectionInfo current = preController.Push(image, isIdle);
            
            if(isIdle)
                return;
            
            hasStarted = true;
            
            preSelectorQueue.Push(current);

        }

        void Finish() {
            //cout << "Pipeline Finish called by " << std::this_thread::get_id() << endl;
            isFinished = true;
            
            aligner->Finish();
            alignerDelayQueue.Flush();
            SelectionInfo last = selectorQueue.GetState();
            stereoConversionQueue.Finish();
            stereoRingBuffer.Push(last);
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

                vector<vector<InputImageP>> rightRings = recorderGraph.SplitIntoRings(rightImages);
                vector<vector<InputImageP>> leftRings = recorderGraph.SplitIntoRings(leftImages);
                leftStore.SaveStitcherInput(leftRings, exposure.GetGains());
                rightStore.SaveStitcherInput(rightRings, exposure.GetGains()); 
            }
        }

        bool HasResults() {
            return leftImages.size() > 0;
        }
                
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            SelectionEdge dummy; 
            return preRecorderGraph.GetEdge(a, b, dummy);
        }
        
        SelectionInfo LastKeyframe() {
            return preSelectorQueue.GetState();
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
