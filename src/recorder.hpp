#include "inputImage.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "monoStitcher.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
#include "ringwiseStitcher.hpp"
#include "checkpointStore.hpp"
#include "lib/control-system-cpp/src/pidController.hpp"

#include "static_timer.hpp"

#include <chrono>

#ifndef OPTONAUT_RECORDER_HEADER
#define OPTONAUT_RECORDER_HEADER

namespace optonaut {
    
    class Recorder {

    private: 

        Mat base;
        Mat baseInv;
        Mat zero;

        shared_ptr<RingwiseStreamAligner> aligner;
        SelectionInfo previous;
        SelectionInfo currentBest;
        SelectionInfo firstOfRing;
        
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
        
        InputImageP GetParentKeyframe(const Mat &extrinsics) {
            return aligner->GetClosestKeyframe(extrinsics);
        }

    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;

        static string tempDirectory;
        static string version;

        static const int stretch = 10;
        
        Recorder(Mat base, Mat zeroWithoutBase, Mat intrinsics, CheckpointStore &leftStore, CheckpointStore &rightStore, int graphConfiguration = RecorderGraph::ModeAll, bool isAsync = true) :
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
            recordedImages(0)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;

            //cout << "Initializing Optonaut Pipe." << endl;
            
            //cout << "Base: " << base << endl;
            //cout << "BaseInv: " << baseInv << endl;
            //cout << "Zero: " << zero << endl;
        
            aligner = shared_ptr<RingwiseStreamAligner>(new RingwiseStreamAligner(recorderGraph, exposure, isAsync));
        }
        
        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
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
            return ConvertFromStitcher(aligner->GetCurrentRotation());
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
        }
        
        void Stitch(const SelectionInfo &a, const SelectionInfo &b, bool discard = false) {
            assert(a.image->IsLoaded());
            assert(b.image->IsLoaded());
            SelectionEdge edge;
            
            exposure.Register(a.image, b.image);

            if(!recorderGraph.GetEdge(a.closestPoint, b.closestPoint, edge))
                return;
           
            StereoImage stereo;
            stereoConverter.CreateStereo(a, b, edge, stereo);

            assert(stereo.valid);

            if(!discard) {
                leftStore.SaveRectifiedImage(stereo.A);
                rightStore.SaveRectifiedImage(stereo.B);
                
                stereo.A->image.Unload();
                stereo.B->image.Unload();

                leftImages.push_back(stereo.A);
                rightImages.push_back(stereo.B);
            }
        }
        
        void ExposureFeed(InputImageP a) {
            InputImageP b = GetParentKeyframe(a->adjustedExtrinsics);
            if(b != NULL && a->id != b->id) {
                exposure.Register(a, b);
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> lt =
                std::chrono::system_clock::now();
        
        static const bool measureTime = true;
        
        void Push(InputImageP image) {
            cout << "Pipeline Push called by " << std::this_thread::get_id() << endl;

            if(isFinished) {
                cout << "Push after finish warning - this could be a racing condition" << endl;
                return;
            }
            
            if(measureTime) {
                auto now = std::chrono::system_clock::now();
                std::cout << "dt=" << std::chrono::duration_cast<std::chrono::microseconds>(now - lt).count() << " mms" << std::endl;
            
                lt = now;
            }
            
            image->originalExtrinsics = base * zero * image->originalExtrinsics.inv() * baseInv;
            
            if(aligner->NeedsImageData() && !image->IsLoaded()) {
                //If the aligner needs image data, pre-load the image.
                image->LoadFromDataRef();
            }
            
            aligner->Push(image);
            
            image->adjustedExtrinsics = aligner->GetCurrentRotation().clone();

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
                if(!image->IsLoaded())
                    image->LoadFromDataRef();
                
                if(current.closestPoint.globalId != currentBest.closestPoint.globalId) {
                    
                    aligner->AddKeyframe(currentBest.image);
                    //Ok, hit that. We can stitch.
                    if(previous.isValid) {
                        Stitch(previous, currentBest);
                        recordedImages++;
                        
                        ExposureFeed(currentBest.image);
                    }
                    if(!firstOfRing.isValid) {
                        firstOfRing = currentBest;
                    } else {
                        if(firstOfRing.closestPoint.ringId != currentBest.closestPoint.ringId) { 
                            Stitch(previous, firstOfRing);
                            firstOfRing = currentBest;
                            previous.isValid = false; //We reached a new ring. Invalidate prev.
                        }
                    }
                
                    previous = currentBest;
                }
                
                currentBest = current;
            }
            
            if(recordedImages == imagesToRecord) {
                isFinished = true;
                Stitch(previous, currentBest);
                Stitch(currentBest, firstOfRing);
            }
        }

        void Finish() {
            //cout << "Pipeline Finish called by " << std::this_thread::get_id() << endl;
            isFinished = true;

            aligner->Finish();
           
            if(HasResults()) {
                if(previous.isValid) {
                    exposure.FindGains();
                }
               
                aligner->Postprocess(leftImages);
                aligner->Postprocess(rightImages);
               
                vector<vector<InputImageP>> rightRings = aligner->SplitIntoRings(rightImages);
                vector<vector<InputImageP>> leftRings = aligner->SplitIntoRings(leftImages);
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
        
        SelectionInfo PreviousPoint() {
            return previous;
        }

        bool IsIdle() {
            return isIdle;
        }
        
        ExposureInfo GetExposureHint() {
            const Mat &current = aligner->GetCurrentRotation();
            
            vector<KeyframeInfo> frames = aligner->GetClosestKeyframes(aligner->GetCurrentRotation(), 2);
            
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
                
                //TODO - rather LERP
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
