#include <algorithm>

#include "image.hpp"
#include "streamAligner.hpp"
#include "monoStitcher.hpp"
#include "imageSelector.hpp"
#include "simpleSphereStitcher.hpp"

#ifndef OPTONAUT_PIPELINE_HEADER
#define OPTONAUT_PIPELINE_HEADER

namespace optonaut {
    class Pipeline {

    private: 

        Mat base;
        Mat baseInv;

        StreamAligner aligner;
        ImageSelector selector; 
        SelectionInfo previous;
        SelectionInfo currentBest;

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

    public: 

        static Mat androidBase;
        static Mat iosBase;
        
        Pipeline(Mat base, Mat intrinsics) : base(base), selector(intrinsics), 
                                          previewImageAvailable(false)
        {
            baseInv = base.inv();
        }

        const Mat &GetOrigin() const {
            return aligner.GetZero();
        }

        const Mat &GetCurrentRotation() const {
            return aligner.GetCurrentRotation();
        }

        const vector<vector<SelectionPoint>> &GetRings() const {
            return selector.GetRings();
        }

        void DisableSelectionPoint(const SelectionPoint &p) {
            return selector.DisableSelectionPoint(p);
        }

        bool IsPreviewImageAvailable() const {
            return previewImageAvailable; 
        }

        ImageP GetPreviewImage() const {
            return lefts.back(); 
        }

        //In: Image with sensor sampled parameters attached. 
        void Push(ImageP image) {
            image->extrinsics = base * image->extrinsics * baseInv;

            aligner.Push(image);
            image->extrinsics = aligner.GetCurrentRotation().clone();

            //Todo - lock to ring. 
            SelectionInfo current = selector.FindClosestSelectionPoint(image);

            previewImageAvailable = false;
      		
            //Remember the closest match for the currently closest point.
            //If we change our closest point, merge the two closest matches
            //for the two past points. 
            
            //cout << "Pushing image: " << image->id << endl;
            if(current.isValid) {

                //cout << "Image valid: " << current.closestPoint.id << endl;

                if(currentBest.isValid && currentBest.closestPoint.id == current.closestPoint.id) {
                    if(currentBest.dist > current.dist) {
                        //Better match for current.
                        currentBest = current;
                        //cout << "Better match" << endl;
                    } 
                } else {
                    //New current point - if we can, merge
                    
                    //cout << "New Point" << endl;
                    if(previous.isValid && currentBest.isValid) {
                        if(selector.AreAdjacent(
                                    previous.closestPoint, 
                                    currentBest.closestPoint)) {
                            StereoImageP stereo = stereoConverter.CreateStereo(previous.image, currentBest.image);
                            //cout << "Doing stereo" << endl;
                            PushLeft(stereo->A);
                            PushRight(stereo->B);

                            previewImageAvailable = true;
                        }
                    }
        
                    previous = currentBest;
                    currentBest = current;               
                }
            }
        }

        StitchingResultP FinishLeft() {
            return leftStitcher.Stitch(lefts, false);
        }       

        StitchingResultP FinishRight() {
            return rightStitcher.Stitch(rights, false);
        }

        bool HasResults() {
            return lefts.size() > 0 && rights.size() > 0;
        }
    };

    
    //Portrait to landscape (use with ios app)
    double iosBaseData[16] = {0, 1, 0, 0,
                             1, 0, 0, 0, 
                             0, 0, 1, 0,
                             0, 0, 0, 1};

    //Landscape L to R (use with android app)
    double androidBaseData[16] = {-1, 0, 0, 0,
                                 0, -1, 0, 0, 
                                 0, 0, 1, 0,
                                 0, 0, 0, 1};

    Mat Pipeline::androidBase(4, 4, CV_64F, androidBaseData);
    Mat Pipeline::iosBase(4, 4, CV_64F, iosBaseData);
}

#endif
