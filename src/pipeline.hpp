#include "image.hpp"
#include "asyncAligner.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "monoStitcher.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "recorderController.hpp"
#include "simpleSphereStitcher.hpp"
#include "imageResizer.hpp"
#include "bundleAligner.hpp"

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

        vector<ImageP> aligned; 

        RStitcher stitcher;

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

        StitchingResultP Finish(vector<ImageP> &images, bool debug = false) {
            auto res = stitcher.Stitch(images, debug);
            resizer.Resize(res->image);
            return res;
        }


    public:

        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;

        static string tempDirectory;
        static string version;

        static bool debug;
        
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
                aligner = shared_ptr<Aligner>(new RingwiseStreamAligner());
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
        
        void Stitch(const SelectionInfo &a, const SelectionInfo &b) {
            assert(a.image->IsLoaded());
            assert(b.image->IsLoaded());
            SelectionEdge edge;
            
            if(!recorderGraph.GetEdge(a.closestPoint, b.closestPoint, edge))
                return;
            
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
            
            image->originalExtrinsics = base * zero * image->originalExtrinsics.inv() * baseInv;
            
            if(aligner->NeedsImageData() && !image->IsLoaded()) {
                //If the aligner needs image data, pre-load the image.
                image->LoadFromDataRef();
            }
            
            aligner->Push(image);
            
            image->adjustedExtrinsics = aligner->GetCurrentRotation().clone();

            if(Pipeline::debug) {
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
                
                if(current.closestPoint.globalId != currentBest.closestPoint.globalId) {
                    //Ok, hit that. We can stitch.
                    if(previous.isValid) {
                        Stitch(previous, currentBest);
                        recordedImages++;
                    }
                    previous = currentBest;
                    
                    
                }
                currentBest = current;
                
            }
            
            //TODO: Something is wrong with the
            //recorder state (off-by-one due to timing?).
            //This is just a quick hack. (-1)
            if(recordedImages == imagesToRecord - 1)
                isFinished = true;
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
            //return Finish(rights, false);
            //Experimental triple stitcher
            auto rings = RingwiseStreamAligner::SplitIntoRings(rights);

            vector<StitchingResultP> stitchedRings;

            for(size_t i = 0; i < rings.size(); i++) {
                auto res = Finish(rings[i], false);
                imwrite("dbg/ring_right_" + ToString(i) + ".jpg", res->image);
                stitchedRings.push_back(res);
      

                if(i > 0) {  
                    const int warp = MOTION_TRANSLATION;
                    Mat affine = Mat::zeros(2, 3, CV_32F);
                    const int iterations = 1000;
                    const double eps = 1e-5;
           
                    affine.at<float>(0, 0) = 1; 
                    affine.at<float>(1, 1) = 1; 
                    affine.at<float>(0, 2) = stitchedRings[0]->corner.x - res->corner.x; 
                    affine.at<float>(1, 2) = stitchedRings[0]->corner.y - res->corner.y;
                    float dx = affine.at<float>(0, 2); 
                    float dy =  affine.at<float>(1, 2); 
                    TermCriteria termination(TermCriteria::COUNT + TermCriteria::EPS, iterations, eps);
                    cout << "ECC initial (" << dx << ", " << dy << ")" << endl; 
                    try {
                        findTransformECC(stitchedRings[0]->image, res->image, affine, warp, termination);
                    } catch (Exception ex) {
                        cout << "ECC couldn't correlate" << endl;
                    }
            
                    cout << "Found Affine: " << affine << endl;
                    dx = affine.at<float>(0, 2); 
                    dy =  affine.at<float>(1, 2); 
                    cout << "ECC corr (" << dx << ", " << dy << ")" << endl; 

                }
            }

            return stitchedRings[0];
        }

        StitchingResultP FinishAligned() {
            return Finish(aligned, false);
        }

        StitchingResultP FinishAlignedDebug() {
            aligner->Postprocess(aligned);
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
