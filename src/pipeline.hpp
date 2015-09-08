#include "image.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "monoStitcher.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
#include "simpleSphereStitcher.hpp"
#include "imageResizer.hpp"

#ifndef OPTONAUT_PIPELINE_HEADER
#define OPTONAUT_PIPELINE_HEADER

namespace optonaut {
    
    class Pipeline {

    private: 

        Mat base;
        Mat baseInv;
        Mat zero;

        shared_ptr<Aligner> aligner;
        SelectionInfo previous;
        SelectionInfo currentBest;
        
        ImageResizer resizer;
        ImageP previewImage;
        MonoStitcher stereoConverter;
        
        vector<ImageP> lefts;
        vector<ImageP> rights; 

        RStitcher leftStitcher;
        RStitcher rightStitcher;

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

        StitchingResultP Finish(vector<ImageP> &images) {
            auto res = leftStitcher.Stitch(images, false);
            resizer.Resize(res->image);
            return res;
        }


    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;

        static string tempDirectory;
        static string version;
        
        Pipeline(Mat base, Mat zeroWithoutBase, Mat intrinsics, int graphConfiguration = RecorderGraph::ModeAll, bool isAsync = true) :
            base(base),
            resizer(graphConfiguration),
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
            cout << "Initializing Optonaut Pipe." << endl;
            
            cout << "Base: " << base << endl;
            cout << "BaseInv: " << baseInv << endl;
            cout << "Zero: " << zero << endl;
        
            baseInv = base.inv();
            zero = zeroWithoutBase;

            if(isAsync) {
                aligner = shared_ptr<Aligner>(new AsyncAligner());
            } else {
                aligner = shared_ptr<Aligner>(new StreamAligner());
            }
            aligner = shared_ptr<Aligner>(new TrivialAligner());
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
            return ConvertFromStitcher(GetPreviewImage()->extrinsics);
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
        
        void Stitch(const SelectionInfo &a, const SelectionInfo &b) {
            assert(a.image->IsLoaded());
            assert(b.image->IsLoaded());
            SelectionEdge edge;
            assert(recorderGraph.HasEdge(a.closestPoint, b.closestPoint, edge));
            
            StereoImage stereo;
            stereoConverter.CreateStereo(a, b, edge, stereo);

            assert(stereo.valid);
            
            CapturePreviewImage(stereo.A);
            
            stereo.A->SaveToDisk();
            stereo.B->SaveToDisk();
            PushLeft(stereo.A);
            PushRight(stereo.B);
        }
        
        //In: Image with sensor sampled parameters attached.
        void Push(ImageP image) {
            
            image->extrinsics = base * zero * image->extrinsics.inv() * baseInv;
            if(aligner->NeedsImageData() && !image->IsLoaded()) {
                //If the aligner needs image data, pre-load the image.
                image->LoadFromDataRef();
            }
            
            aligner->Push(image);
            image->extrinsics = aligner->GetCurrentRotation().clone();

            previewImageAvailable = false;
            
            if(!controller.IsInitialized())
                controller.Initialize(image->extrinsics);
      		
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
                    //Ok, hit that. We can stitch.
                    if(previous.isValid) {
                        Stitch(previous, currentBest);
                        cout << "Remove edge" << endl;
                        
                        recordedImages++;
                    }
                    previous = currentBest;
                }
            
                currentBest = current;
            }
            
            if(recordedImages == imagesToRecord)
                isFinished = true;
        }
                
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            SelectionEdge dummy; 
            return recorderGraph.HasEdge(a, b, dummy);
        }
        
        SelectionInfo CurrentPoint() {
            return currentBest;
        }
        
        SelectionInfo PreviousPoint() {
            return previous;
        }

        StitchingResultP FinishLeft() {
            return Finish(lefts);
        }       

        StitchingResultP FinishRight() {
            return Finish(rights);
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
