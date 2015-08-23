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

        AsyncAligner aligner;
        ImageSelector selector; 
        SelectionInfo previous;
        SelectionInfo currentBest;
        ImageResizer resizer;
        PipelineState state;

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

        //int selectorConfiguration; 

    public: 

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;
        
        Pipeline(Mat base, Mat zeroWithoutBase, Mat intrinsics, int selectorConfiguration = ImageSelector::ModeCenter) : 
            base(base),
            selector(intrinsics, selectorConfiguration),
            resizer(selectorConfiguration),
            previewImageAvailable(false)
        {
            baseInv = base.inv();
            zero = zeroWithoutBase;
            /*
            cout << "Initializing Optonaut Pipe." << endl;
            
            cout << "Base: " << base << endl;
            cout << "BaseInv: " << baseInv << endl;
            cout << "Zero: " << zero << endl;
            */
        }

        //Methods already coordinates in input base. 
        Mat GetOrigin() const {
            return baseInv * zero * base;
        }

        Mat GetCurrentRotation() const {
            return (zero.inv() * baseInv * aligner.GetCurrentRotation() * base).inv();
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
            return lefts.back();
        }
        
        Mat GetPreviewRotation() {
            return (zero.inv() * baseInv * GetPreviewImage()->extrinsics * base).inv();
        }

        void Dispose() {
            aligner.Dispose();
        }

        //In: Image with sensor sampled parameters attached. 
        void Push(ImageP image) {
        
            image->extrinsics = base * zero * image->extrinsics.inv() * baseInv;
            if(aligner.NeedsImageData()) {
                //If the aligner needs image data, pre-load the image. 
                image->Load();
            }
            aligner.Push(image);
            image->extrinsics = aligner.GetCurrentRotation().clone();

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
                        currentBest.image->Load(); //Need to get image contents now. 
                        //cout << "Better match" << endl;
                    } 
                } else {
                    //New current point - if we can, merge
                    
                    //cout << "New Point" << endl;
                    if(previous.isValid && currentBest.isValid) {
                        if(AreAdjacent(
                                    previous.closestPoint, 
                                    currentBest.closestPoint)) {
                            
                            StereoImageP stereo = stereoConverter.CreateStereo(previous.image, currentBest.image);

                            //cout << "Doing stereo" << endl;
                            PushLeft(stereo->A);
                            PushRight(stereo->B);

                            previewImageAvailable = true;
                        } else {
                            //cout << "Images not adjacent" << endl;
                        }
                        selector.DisableAdjacency(previous.closestPoint, currentBest.closestPoint);
                    }
        
                    previous = currentBest;
                    currentBest = current;               
                }
            }
        }
        
        PipelineState GetState() {
            return state;
        }
        
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            return selector.AreAdjacent(a, b);
        }
        
        SelectionInfo CurrentPoint() {
            return currentBest;
        }
        
        SelectionInfo PreviousPoint() {
            return previous;
        }

        StitchingResultP FinishLeft() {
            auto left = leftStitcher.Stitch(lefts, false);
            resizer.Resize(left->image);
            return left;
        }       

        StitchingResultP FinishRight() {
            auto right = rightStitcher.Stitch(rights, false);
            resizer.Resize(right->image);
            return right;
        }

        bool HasResults() {
            return lefts.size() > 0 && rights.size() > 0;
        }
    };

    
    //Portrait to landscape (use with ios app)
    double iosBaseData[16] = {
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    //Landscape L to R (use with android app)
    double androidBaseData[16] = {
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    //Base picked from exsiting data - we might find something better here. 
    double iosZeroData[16] = {
        0, 0, 1, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1
    };


    Mat Pipeline::androidBase(4, 4, CV_64F, androidBaseData);
    Mat Pipeline::iosBase(4, 4, CV_64F, iosBaseData);
    Mat Pipeline::iosZero = Mat(4, 4, CV_64F, iosZeroData);
}

#endif
