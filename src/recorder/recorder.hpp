#include "../io/inputImage.hpp"
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

#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
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
        
        RecorderGraphGenerator generator;
        RecorderGraph recorderGraph;
        RecorderController controller;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;

        ReduceProcessor<SelectionInfo, SelectionInfo> selectorQueue;
        RingProcessor<SelectionInfo> monoQueue;
        QueueProcessor<InputImageP> alignerQueue;
        AsyncQueue<StereoPair> stereoProcessor;

        vector<SelectionInfo> firstRing;
        vector<InputImageP> firstRingImagePool;
        
        STimer monoTimer;
        STimer pipeTimer;
        
        InputImageP last;
        
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
                CheckpointStore &leftStore, CheckpointStore &rightStore, CheckpointStore &commonStore,
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
            generator(),
            recorderGraph(generator.Generate(intrinsics, graphConfiguration)),
            controller(recorderGraph),
            imagesToRecord(recorderGraph.Size()),
            recordedImages(0),
            selectorQueue(
                std::bind(&Recorder::SelectBetterMatch,
                    this, placeholders::_1,
                    placeholders::_2), SelectionInfo()),
            monoQueue(1,
                std::bind(&Recorder::ForwardToStereoQueue, 
                    this, placeholders::_1,
                    placeholders::_2),
                std::bind(&Recorder::FinishImage, 
                    this, placeholders::_1)),
            alignerQueue(60, 
                std::bind(&Recorder::ApplyAlignment, 
                    this, placeholders::_1)),
            stereoProcessor(
                std::bind(&Recorder::StitchImages, 
                    this, placeholders::_1))
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

        void ForwardToStereoQueue(const SelectionInfo &a, const SelectionInfo &b) {
            StereoPair pair;
            pair.a = a;
            pair.b = b;
            stereoProcessor.Push(pair);
        }
        
        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
        }
        
        void ConvertToStitcher(const Mat &in, Mat &out) const {
            out = (base * zero * in.inv() * baseInv);
        }
       
        //TODO: Rename/Remove all ball methods.  
        Mat GetBallPosition() const {
            return ConvertFromStitcher(controller.GetBallPosition());
        }
        
        double GetDistanceToBall() const {
            return controller.GetError();
        }
        
        const Mat &GetAngularDistanceToBall() const {
            return controller.GetErrorVector();
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
            for(auto ring : recorderGraph.GetRings())
                for(auto point : ring) {
                    SelectionPoint n;
                    n.globalId = point.globalId;
                    n.ringId = point.ringId;
                    n.localId = point.localId;
                    n.extrinsics = ConvertFromStitcher(point.extrinsics);
                    
                    converted.push_back(n);
            }
            //cout << "returning " << converted.size() << " rings " << endl;
            return converted;
        }

        void Dispose() {
            //cout << "Pipeline Dispose called by " << std::this_thread::get_id() << endl;
            aligner->Dispose();
            stereoProcessor.Dispose();
        }
        
        void FinishImage(const SelectionInfo &fin) {
            ExposureFeed(fin.image);
        }   
        
        void StitchImages(const StereoPair &pair) {
            cout << "Stitch" << endl;
            
            if(!pair.a.image->image.IsLoaded()) {
                pair.a.image->image.Load();
            }
            if(!pair.b.image->image.IsLoaded()) {
                pair.b.image->image.Load();
            }
            
            assert(pair.a.image->IsLoaded());
            assert(pair.b.image->IsLoaded());

            monoTimer.Reset();
            
            if(exposureEnabled)
                exposure.Register(pair.a.image, pair.b.image);
            
            monoTimer.Tick("Exposure Comp");
            
            StereoImage stereo;
            stereoConverter.CreateStereo(pair.a, pair.b, stereo);

            assert(stereo.valid);
            monoTimer.Tick("Stereo Conv");

            leftStore.SaveRectifiedImage(stereo.A);
            rightStore.SaveRectifiedImage(stereo.B);
            
            stereo.A->image.Unload();
            stereo.B->image.Unload();

            leftImages.push_back(stereo.A);
            rightImages.push_back(stereo.B);
            monoTimer.Tick("Stereo Store");
        }
        
        void ExposureFeed(InputImageP a) {
            
            if(!exposureEnabled)
                return;
            
            InputImageP b = GetParentKeyframe(a->adjustedExtrinsics);
            if(b != NULL && a->id != b->id) {
                exposure.Register(a, b);
            }
        }
        
        void ForwardToMonoQueue(const SelectionInfo &in, bool poolOnly = false) {
            if(!firstRingFinished) {
                if(!poolOnly && 
                    (firstRing.size() == 0 
                        || firstRing.back().closestPoint.globalId != 
                        in.closestPoint.globalId)) {
                    firstRing.push_back(in);    
                }
                firstRingImagePool.push_back(in.image);
                commonStore.SaveStitcherTemporaryImage(in.image->image);
                in.image->image.Unload();
            } else if(!poolOnly) {
                ForwardToMonoQueueEx(in);
            }
        }

        void FinishFirstRing() {
            AssertM(!firstRingFinished, "First ring has not been closed");
            firstRingFinished = true;

            PairwiseCorrelator corr;
            firstRingImagePool.back()->image.Load();
            firstRingImagePool.front()->image.Load();
            auto result = corr.Match(firstRingImagePool.back(), firstRingImagePool.front()); 
            size_t n = firstRingImagePool.size();

            cout << "Y horizontal angular offset: " << result.horizontalAngularOffset << endl; 

            for(size_t i = 0; i < n; i++) {
                double ydiff = result.horizontalAngularOffset * 
                    (1.0 - ((double)i) / ((double)n));
                Mat correction;
                CreateRotationY(ydiff, correction);
                firstRingImagePool[i]->adjustedExtrinsics = correction * 
                    firstRingImagePool[i]->adjustedExtrinsics;

                cout << "Applying correction of " << ydiff << " to image " << firstRingImagePool[i]->id << endl;
            }

            //Debug 
            //SimpleSphereStitchrer stitcher;
            //auto scene = stitcher.Stitch(images);
            //imwrite("dbg/extracted_ring.jpg", scene->image.data);
    
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
                cout << "Selection Point " << firstRing[i].closestPoint.globalId
                    << " reassigned image " << firstRing[i].image->id << " to " 
                    << firstRingImagePool[picked]->id << " with dist: " 
                    << distCur << endl;
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
            recordedImages++;
            
            if(recordedImages % 2 == 0) {
                //Save some memory by sparse keyframing.
                if(!in.image->image.IsLoaded()) {
                    in.image->image.Load();
                }
                aligner->AddKeyframe(in.image);
            }

            monoQueue.Push(in);
        }

        void FlushMonoQueue() {
            monoQueue.Flush();
        }

        SelectionInfo SelectBetterMatch(const SelectionInfo &best, const SelectionInfo &in) {
            //Init case
            if(!best.isValid)
                return in;

            //Invalid case
            if(!in.isValid)
                return best;
          
            if(best.closestPoint.globalId != in.closestPoint.globalId) {
                //We will not get a better match. Can forward best.  
                ForwardToMonoQueue(best);
                //ForwardToMonoQueue(in, true);
                //We Finished a ring. Flush. 
                if(best.closestPoint.ringId != in.closestPoint.ringId) { 
                    if(!firstRingFinished) {
                        FinishFirstRing();
                    }
                    FlushMonoQueue();
                }

            } else {
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

            if(!controller.IsInitialized())
                controller.Initialize(image->adjustedExtrinsics);
            
            SelectionInfo current = controller.Push(image, isIdle);

            if(isIdle)
                return;
           
            selectorQueue.Push(current); 
            
            if(recordedImages == imagesToRecord) {
                //Special Case for last image. 
                SelectionInfo last = selectorQueue.GetState();
                monoQueue.Push(last);
                FlushMonoQueue();
                isFinished = true;
            }
        }
    
        void Push(InputImageP image) {
            //cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;
            
            last = image;

            if(isFinished) {
                cout << "Push after finish warning - this could be a racing condition" << endl;
                return;
            }
            
            //pipeTimer.Tick("Push");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
            image->adjustedExtrinsics = image->originalExtrinsics;
           
            //Load all the images.  
            if(!image->IsLoaded()) {
                image->LoadFromDataRef();
            }
            
            aligner->Push(image);
            alignerQueue.Push(image);
            //ApplyAlignment(image);
        }

        void Finish() {
            //cout << "Pipeline Finish called by " << std::this_thread::get_id() << endl;
            isFinished = true;

            aligner->Finish();
            alignerQueue.Flush();
            stereoProcessor.Finish();
           
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
            return recorderGraph.GetEdge(a, b, dummy);
        }
        
        SelectionInfo CurrentPoint() {
            return selectorQueue.GetState();
        }
        
        bool IsIdle() {
            return isIdle;
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
