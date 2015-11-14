#include "../io/inputImage.hpp"
#include "../io/checkpointStore.hpp"
#include "../stereo/monoStitcher.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/queueProcessor.hpp"
#include "../common/asyncQueueWorker.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
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
        SelectionInfo currentBest;
        
        CheckpointStore leftStore;
        CheckpointStore rightStore;
        
        vector<InputImageP> leftImages;
        vector<InputImageP> rightImages;

        ExposureCompensator exposure;
        
        MonoStitcher stereoConverter;

        bool isIdle;
        bool isFinished;
        
        RecorderGraphGenerator generator;
        RecorderGraph recorderGraph;
        RecorderController controller;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;

        RingProcessor<SelectionInfo> monoQueue;
        QueueProcessor<InputImageP> alignerQueue;
        AsyncQueue<StereoPair> stereoProcessor;
        
        STimer monoTimer;
        STimer pipeTimer;
        
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
                CheckpointStore &leftStore, CheckpointStore &rightStore, 
                int graphConfiguration = RecorderGraph::ModeAll, 
                bool isAsync = true) :
            base(base),
            leftStore(leftStore),
            rightStore(rightStore),
            stereoConverter(),
            isIdle(false),
            isFinished(false),
            generator(),
            recorderGraph(generator.Generate(intrinsics, graphConfiguration)),
            controller(recorderGraph),
            imagesToRecord(recorderGraph.Size()),
            recordedImages(0),
            monoQueue(1,
                    [this] (const SelectionInfo &a, const SelectionInfo &b) {
                        StereoPair pair;
                        pair.a = a;
                        pair.b = b;
                        stereoProcessor.Push(pair);
                    },
                std::bind(&Recorder::FinishImage, 
                    this, placeholders::_1)),
            alignerQueue(30, std::bind(&Recorder::ApplyAlignment, 
                this, placeholders::_1)),
            stereoProcessor([this] (const StereoPair &a) { StitchImages(a); })
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
        
        Mat ConvertFromStitcher(const Mat &in) const {
            assert(false); //Dangerous to use. Allocation error. 
            return (zero.inv() * baseInv * in * base).inv();
        }
        
        void ConvertToStitcher(const Mat &in, Mat &out) const {
            out = (base * zero * in.inv() * baseInv);
        }
       
        //TODO: Rename all ball methods.  
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
            assert(false); //Wrong semantics.
            return ConvertFromStitcher(aligner->GetCurrentBias());
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
        
        void FinishImage(SelectionInfo &fin) {
            ExposureFeed(fin.image);
        }   
        
        void StitchImages(const StereoPair &pair) {
            cout << "Stitch" << endl;

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

        void ApplyAlignment(InputImageP image) {

            cout << "Apply correction to image " << image->id << endl;
            cout << aligner->GetCurrentBias() << endl;

            image->adjustedExtrinsics = aligner->GetCurrentBias() * image->originalExtrinsics;

            if(!controller.IsInitialized())
                controller.Initialize(image->adjustedExtrinsics);
            
            SelectionInfo current = controller.Push(image, isIdle);

            if(isIdle)
                return;
            
            if(!currentBest.isValid) {
                //Initialization. 
                currentBest = current;
            }

            if(current.isValid) {
                if(!current.image->IsLoaded())
                    current.image->LoadFromDataRef();
                
                if(current.closestPoint.globalId != 
                        currentBest.closestPoint.globalId) {
            cout << "Valid Frame:  " << image->id << endl;
                    
                    if(recordedImages % 2 == 0) {     
                        //Save some memory by sparse keyframing. 
                        aligner->AddKeyframe(currentBest.image);
                    }
                    //Ok, hit that. We can stitch.
                    monoQueue.Push(currentBest);
                    recordedImages++;
                    
                    if(current.closestPoint.ringId != 
                            currentBest.closestPoint.ringId) { 
                        monoQueue.Flush();
                    }
                }
                
                currentBest = current;
            }
            
            if(recordedImages == imagesToRecord) {
                monoQueue.Push(currentBest);
                monoQueue.Flush();
                isFinished = true;
            }
        }
    
        void Push(InputImageP image) {
            //cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;

            if(isFinished) {
                cout << "Push after finish warning - this could be a racing condition" << endl;
                return;
            }
            
            //pipeTimer.Tick("Push");
            
            ConvertToStitcher(image->originalExtrinsics, image->originalExtrinsics);
            image->adjustedExtrinsics = image->originalExtrinsics;
            
            if(aligner->NeedsImageData() && !image->IsLoaded()) {
                //If the aligner needs image data, pre-load the image.
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
            return currentBest;
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
