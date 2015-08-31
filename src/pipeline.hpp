#include "image.hpp"
#include "asyncAligner.hpp"
#include "monoStitcher.hpp"
#include "imageSelector.hpp"
#include "simpleSphereStitcher.hpp"
#include "imageResizer.hpp"

#ifndef OPTONAUT_PIPELINE_HEADER
#define OPTONAUT_PIPELINE_HEADER

namespace optonaut {
    
    struct PipelineState {
        bool isOnRing;
        
        PipelineState() : isOnRing(false) {
            
        }
    };
    
    class Pipeline {

    private: 

        Mat base;
        Mat baseInv;
        Mat zero;

        shared_ptr<Aligner> aligner;
        ImageSelector selector; 
        SelectionInfo previous;
        SelectionInfo currentBest;
        ImageResizer resizer;
        PipelineState state;
        ImageP previewImage;

        MonoStitcher stereoConverter;

        vector<ImageP> lefts;
        vector<ImageP> rights; 

        RStitcher leftStitcher;
        RStitcher rightStitcher;

        bool previewImageAvailable;

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
        
        Pipeline(Mat base, Mat zeroWithoutBase, Mat intrinsics, int selectorConfiguration = ImageSelector::ModeAll, bool isAsync = true) :
            base(base),
            selector(intrinsics, selectorConfiguration),
            resizer(selectorConfiguration),
            previewImageAvailable(false)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;


            if(isAsync) {
                aligner = shared_ptr<Aligner>(new AsyncAligner());
            } else {
                aligner = shared_ptr<Aligner>(new StreamAligner());
            }
            cout << "Initializing Optonaut Pipe." << endl;
            
            cout << "Base: " << base << endl;
            cout << "BaseInv: " << baseInv << endl;
            cout << "Zero: " << zero << endl;
        }

        //Methods already coordinates in input base. 
        Mat GetOrigin() const {
            return baseInv * zero * base;
        }

        Mat GetCurrentRotation() const {
            return (zero.inv() * baseInv * aligner->GetCurrentRotation() * base).inv();
        }

        vector<SelectionPoint> GetSelectionPoints() const {
            vector<SelectionPoint> converted;
            for(auto ring : selector.GetRings())
                for(auto point : ring) {
                    SelectionPoint n;
                    n.id = point.id;
                    n.ringId = point.ringId;
                    n.localId = point.localId;
                    n.enabled = point.enabled;
                    n.extrinsics = (zero.inv() * baseInv * point.extrinsics * base).inv();
                    
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
            return (zero.inv() * baseInv * GetPreviewImage()->extrinsics * base).inv();
        }

        void Dispose() {
            aligner->Dispose();
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

            //Todo - lock to ring. 
            SelectionInfo current = selector.FindClosestSelectionPoint(image);

            //cout << "image " << image->id << " closest to " << current.closestPoint.id << ", dist: " << current.dist << ", ring: " << current.closestPoint.ringId << endl;

            previewImageAvailable = false;
      		
            //Remember the closest match for the currently closest point.
            //If we change our closest point, merge the two closest matches
            //for the two past points. 
            
            //cout << "Pushing image: " << image->id << endl;
            if(current.isValid) {
                
                state.isOnRing = true;

                //cout << "Image valid: " << current.closestPoint.id << endl;

                if(currentBest.isValid && currentBest.closestPoint.id == current.closestPoint.id) {
                    if(currentBest.dist > current.dist) {
                        //Better match for current.
                        currentBest = current;
                        

                        if(!currentBest.image->IsLoaded())
                            currentBest.image->LoadFromDataRef(); //Need to get image contents now. 
                        //cout << "Better match" << endl;
                    } 
                } else {
                    //New current point - if we can, merge
                    
                    //cout << "New Point" << endl;
                    if(previous.isValid && currentBest.isValid) {
                        SelectionEdge edge; 
                        if(selector.AreAdjacent(
                                    previous.closestPoint, 
                                    currentBest.closestPoint, edge)) {

                            assert(previous.image->IsLoaded());
                            assert(currentBest.image->IsLoaded());


                            StereoTarget target;
                            target.center = edge.roiCenter;

                            for(int i = 0; i < 4; i++) {
                                target.corners[i] = edge.roiCorners[i];
                            }
                        
                            currentBest.image->offset = currentBest.image->extrinsics.inv() * currentBest.closestPoint.extrinsics;
                            previous.image->offset = previous.image->extrinsics.inv() * previous.closestPoint.extrinsics;
                            
                            StereoImageP stereo = stereoConverter.CreateStereo(previous.image, currentBest.image, target);
                            
                            if(stereo->valid) {
                                //cout << "Doing stereo" << endl;
                                
                                previewImage = ImageP(new Image(*stereo->A));
                                previewImage->img = stereo->A->img.clone();
                                
                                stereo->A->SaveToDisk();
                                stereo->B->SaveToDisk();
                                PushLeft(stereo->A);
                                PushRight(stereo->B);

                                previewImageAvailable = true;
                            }
                        } else {
                            //cout << "Images not adjacent" << endl;
                        }
                        selector.DisableAdjacency(previous.closestPoint, currentBest.closestPoint);
                    }
        
                    previous = currentBest;
                    currentBest = current;
                    
                    if(!currentBest.image->IsLoaded())
                        currentBest.image->LoadFromDataRef();
                }
            }
        }
        
        PipelineState GetState() {
            return state;
        }
        
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            SelectionEdge dummy; 
            return selector.AreAdjacent(a, b, dummy);
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
    };    
}

#endif
