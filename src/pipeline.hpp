#include "image.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "monoStitcher.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
#include "ringwiseStitcher.hpp"

#include <chrono>

#ifndef OPTONAUT_PIPELINE_HEADER
#define OPTONAUT_PIPELINE_HEADER

namespace optonaut {
    
    class Pipeline {

    private: 

        Mat base;
        Mat baseInv;
        Mat zero;

        shared_ptr<RingwiseStreamAligner> aligner;
        SelectionInfo previous;
        SelectionInfo currentBest;
        SelectionInfo firstOfRing;

        ExposureCompensator exposure;
        
        ImageP previewImage;
        MonoStitcher stereoConverter;
        
        vector<ImageP> lefts;
        vector<ImageP> rights;

        vector<ImageP> aligned; 

        RingwiseStitcher stitcher;

        bool previewImageAvailable;
        bool isIdle;
        bool previewEnabled;
        bool isFinished;
        
        RecorderGraphGenerator generator;
        RecorderGraph recorderGraph;
        RecorderController controller;
        
        uint32_t imagesToRecord;
        uint32_t recordedImages;

        
        void PushLeft(ImageP left) {
            lefts.push_back(left);
        }

        void PushRight(ImageP right) {
            rights.push_back(right);
        }

        StitchingResultP Finish(vector<ImageP> &images, bool debug = false, string debugName = "") {
            aligner->Postprocess(images);
            auto rings = RingwiseStreamAligner::SplitIntoRings(images, recorderGraph);
            
            StitchingResultP res;
            //Change the following line for HDR. 
            vector<double> evs = { 0.4 };
            vector<Mat> hdrs;
            
            for(size_t i = 0; i < evs.size(); i++) {
                auto lres = stitcher.Stitch(rings, exposure, evs[i], debug, debugName);
                if(i == 0) {
                    res = lres;
                }
                
                hdrs.push_back(lres->image);
            }
            if(hdrs.size() > 1) {
                Mat fusion;
                Ptr<MergeMertens> merge = createMergeMertens();
                merge->process(hdrs, fusion);

                fusion.convertTo(res->image, CV_8U, 255);
            }
            return res;
        }


    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;

        static string tempDirectory;
        static string version;

        static bool debug;
        static const int stretch = 10;
        
        Pipeline(Mat base, Mat zeroWithoutBase, Mat intrinsics, int graphConfiguration = RecorderGraph::ModeAll, bool isAsync = true) :
            base(base),
            stereoConverter(),
            previewImageAvailable(false),
            isIdle(false),
            previewEnabled(true),
            isFinished(false),
            generator(),
            recorderGraph(generator.Generate(intrinsics, graphConfiguration)),
            controller(recorderGraph),
            imagesToRecord(recorderGraph.Size()),
            recordedImages(0)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;

            cout << "Initializing Optonaut Pipe." << endl;
            
            cout << "Base: " << base << endl;
            cout << "BaseInv: " << baseInv << endl;
            cout << "Zero: " << zero << endl;
        
            aligner = shared_ptr<RingwiseStreamAligner>(new RingwiseStreamAligner(recorderGraph, exposure, isAsync));

            stitcher = RingwiseStitcher(4096, 4096);
        }
        
        void SetPreviewImageEnabled(bool enabled) {
            previewEnabled = enabled;
        }
        
        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
        }
        
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
                    n.enabled = point.enabled;
                    n.extrinsics = ConvertFromStitcher(point.extrinsics);
                    
                    converted.push_back(n);
            }
            //cout << "returning " << converted.size() << " rings " << endl;
            return converted;
        }

        bool IsPreviewImageAvailable() const {
            return previewImageAvailable; 
        }

        ImageP GetPreviewImage() const {
            return previewImage;
        }
        
        Mat GetPreviewRotation() {
            return ConvertFromStitcher(GetPreviewImage()->adjustedExtrinsics);
        }

        void Dispose() {
            aligner->Dispose();
        }
        
        void CapturePreviewImage(const ImageP img) {
            if(previewEnabled) {
                previewImage = ImageP(new Image(*img));
                previewImage->img = img->img.clone();
                
                previewImageAvailable = true;
            }
        }

        void Stitch(const SelectionInfo &a, const SelectionInfo &b, bool discard = false) {
            cout << "stitching " << a.image->id << " and " << b.image->id << endl;
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
                CapturePreviewImage(stereo.A);
                
                stereo.A->SaveToDisk();
                stereo.B->SaveToDisk();
                PushLeft(stereo.A);
                PushRight(stereo.B);
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> lt = std::chrono::system_clock::now();
        
        static const bool measureTime = false;
        
        void Push(ImageP image) {
            
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

            if(Pipeline::debug && image->id % stretch == 0) {
                if(!image->IsLoaded())
                    image->LoadFromDataRef();
                aligned.push_back(image);
            }

            previewImageAvailable = false;
            
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

                cout << "valid " << current.image->id << endl;
                
                if(current.closestPoint.globalId != currentBest.closestPoint.globalId) {
                    
                    aligner->AddKeyframe(currentBest.image);
                    //Ok, hit that. We can stitch.
                    if(previous.isValid) {
                        Stitch(previous, currentBest);
                        recordedImages++;
                
                        cout << "recorded: " << recordedImages << "/" << imagesToRecord << endl;
                    }
                    if(!firstOfRing.isValid) {
                        firstOfRing = currentBest;
                    } else {
                        if(firstOfRing.closestPoint.ringId != currentBest.closestPoint.ringId) { 
                            Stitch(previous, firstOfRing, true);
                            firstOfRing = currentBest;
                            previous.isValid = false; //We reached a new ring. Invalidate prev.
                        }
                    }
                    
                    ImageP closestKeyframe = aligner->GetClosestKeyframe(currentBest.image->adjustedExtrinsics);
                    if(closestKeyframe != NULL) {
                        exposure.Register(currentBest.image, closestKeyframe);
                    }
                    previous = currentBest;
                }
                currentBest = current;
            }
            
            if(recordedImages == imagesToRecord) {
                cout << "Finished" << endl;
                isFinished = true;
                Stitch(currentBest, firstOfRing, true);
            }
        }

        void Finish() {
            isFinished = true;
            if(previous.isValid) {
                Stitch(previous, currentBest);
                exposure.FindGains();
            }
            
            aligner->Finish();
            //exposure.PrintCorrespondence();
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

        StitchingResultP FinishLeft() {
            return Finish(lefts, false);
        }

        StitchingResultP FinishRight() {
            return Finish(rights, false);
        }

        StitchingResultP FinishAligned() {
            return Finish(aligned, false);
        }

        StitchingResultP FinishAlignedDebug() {
            return Finish(aligned, true);
        }

        bool HasResults() {
            return lefts.size() > 0 && rights.size() > 0;
        }
        
        bool IsIdle() {
            return isIdle;
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
