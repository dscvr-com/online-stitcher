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
        RecorderGraphGenerator generator;
        
        ImageResizer resizer;
        ImageP previewImage;
        MonoStitcher stereoConverter;
        RecorderGraph recorderGraph;
        RecorderController controller;

        vector<ImageP> lefts;
        vector<ImageP> rights; 

        RStitcher leftStitcher;
        RStitcher rightStitcher;

        bool previewImageAvailable;
        bool isIdle;
        
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
            generator(),
            resizer(graphConfiguration),
            previewImageAvailable(false),
            isIdle(false),
            recorderGraph(generator.Generate(intrinsics, graphConfiguration)),
            controller(recorderGraph)
        {
            cout << "Initializing Optonaut Pipe." << endl;
            
            cout << "Base: " << base << endl;
            cout << "BaseInv: " << baseInv << endl;
            cout << "Zero: " << zero << endl;
        
            baseInv = base.inv();
            zero = zeroWithoutBase;

            aligner = shared_ptr<Aligner>(new TrivialAligner());
            //if(isAsync) {
            //    aligner = shared_ptr<Aligner>(new AsyncAligner());
            //} else {
            //    aligner = shared_ptr<Aligner>(new StreamAligner());
            //}
        }
        
        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
        }
        
        Mat GetBallPosition() const {
            return ConvertFromStitcher(controller.GetBallPosition());
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
            
            previewImage = ImageP(new Image(*img));
            previewImage->img = img->img.clone();
            
            previewImageAvailable = true;
        }
        
        void Stitch(const SelectionInfo &a, const SelectionInfo &b) {
            assert(a.image->IsLoaded());
            assert(b.image->IsLoaded());
            SelectionEdge edge;
            assert(recorderGraph.HasEdge(previous.closestPoint, currentBest.closestPoint, edge));
            
            StereoImage stereo;
            stereoConverter.CreateStereo(previous, currentBest, edge, stereo);

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
            
            if(isIdle)
                return;
            
            if(!controller.IsInitialized())
                controller.Initialize(image->extrinsics);
      		
            SelectionInfo current = controller.Push(image);
            
            if(!currentBest.isValid) {
                //Initialization. 
                currentBest = current;
            }
            
            if(current.isValid) {
                if(!image->IsLoaded())
                    image->LoadFromDataRef();
                
                if(current.closestPoint.globalId != currentBest.closestPoint.globalId) {
                    //Ok, hit that. We can stitch.
                    if(previous.isValid)
                        Stitch(previous, currentBest);
                    previous = currentBest;
                    
                    
                }
                currentBest = current;
                
            }
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
        
        void SetIdle(bool isIdle) {
            this->isIdle = isIdle;
        }
    };    
}

#endif
