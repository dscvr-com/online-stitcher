#include <opencv2/opencv.hpp>
#include "../common/image.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
#include "../io/checkpointStore.hpp"
#include "../io/inputImage.hpp"
#include "../io/io.hpp"
#include "../stereo/monoStitcher.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/functional.hpp"
#include "../recorder/recorderGraph.hpp"


#include "multiringStitcher.hpp"
#include "stitcher.hpp"
#include "../recorder/recorderGraph.hpp"
#include "../recorder/ringCloser.hpp"
#include "../recorder/recorderGraphGenerator.hpp"
#include "../recorder/iterativeBundleAligner.hpp"
#include "../minimal/imagePreperation.hpp"

#include <chrono>
#include <string>


#ifndef OPTONAUT_GLOBAL_ALIGNMENT_HEADER
#define OPTONAUT_GLOBAL_ALIGNMENT_HEADER
using namespace cv;
using namespace std;

namespace optonaut {

    /*
     * Wrapper for the optonaut stitching part.
     */
    class GlobalAlignment {

    private:
        static const bool debug = false;
        static const bool fillMissingImages = false;

        CheckpointStore &imageStore;
        CheckpointStore &leftStore;
        CheckpointStore &rightStore;
        RecorderGraphGenerator generator;
    public:

        GlobalAlignment(CheckpointStore &imageStore, CheckpointStore &leftStore,
                        CheckpointStore &rightStore ) :
            imageStore(imageStore), leftStore(leftStore), rightStore(rightStore), generator() {
                AssertFalseInProduction(debug);

                // If you want to fill missing images, please configure a dynamic path
                // suitable for production in the AddDummyImages method.
                // Then remove this assert.
                AssertFalseInProduction(fillMissingImages);
        }

        // Loads mini images from the source, then unloads the original. 
        void MinifyImages(vector<InputImageP> &images, int downsample = 2) {
           AssertGT(downsample, 0);

           int  counter = 0;
           for(auto img : images) {
                std::string source = img->image.source;
                if(!img->image.IsLoaded()) {
                    img->image.Load();
                    AssertM(img->image.data.cols != 0, "Image loaded successfully");
                }

                cv::Mat small;

                pyrDown(img->image.data, small);

                for(int i = 1; i < downsample; i++) {
                    pyrDown(small, small);
                }

                counter++;
                img->image = Image(small);
                img->image.source = source;

            }
        }

        // This is a copy of getVerticalFov
        double GetVerticalFov ( const Mat &intrinsics ) {
             double h = intrinsics.at<double>(1,2);
             double f = intrinsics.at<double>(0,0);
             return 2 * atan2(h,f);
        }

        // Runs the Actual Operation. 
        // Beautiful Spagetthi Code. 
        void Finish() {

            // #### Step 1: Load everything and prepare graphs and stuff. 
            STimer timer;

            Log << "Finish";

            vector<std::vector<InputImageP>> loadedRings;
            BiMap<size_t, uint32_t> imagesToTargets, d;
            map<size_t, double> gains;
            
            imageStore.LoadStitcherInput(loadedRings, gains);

            vector<InputImageP> inputImages = fun::flat(loadedRings);
            cv::Size originalSize = inputImages[0]->image.size();

            Log << "Images Loaded.";

            Mat intrinsics;
            intrinsics = inputImages[0]->intrinsics;
            int graphConfiguration = 0;

            if (loadedRings.size() == 1) {
                graphConfiguration =  RecorderGraph::ModeCenter;
                Log << "Creating centered recorder graph";
            } else if (loadedRings.size() == 3)  {
                graphConfiguration =  RecorderGraph::ModeTruncated;
                Log << "Creating three ring recorder graph";
            }

            Log << "Using Intrinsics " << intrinsics;

            RecorderGraph recorderGraph = generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityNormal, 0, 8);

            vector<InputImageP> best = recorderGraph.SelectBestMatches(inputImages, imagesToTargets, false);
            Log << "Pre-Alignment, found " << best.size() << "/" << recorderGraph.Size() << "/" << inputImages.size();

            if(debug) {
                SimpleSphereStitcher debugger;
                minimal::ImagePreperation::LoadAllImages(best);
                imwrite("dbg/aligner_input.jpg", debugger.Stitch(best, false, true)->image.data);
            }

            timer.Tick("Init'ed recorder graph and found best matches");

            // #### Step 2: Load mini images and run bundle adjustment

            int downsample = 3;
            MinifyImages(best, downsample);

            timer.Tick("Loaded mini images");

            vector<vector<InputImageP>> rings = recorderGraph.SplitIntoRings(best);
            size_t k = rings.size() / 2;

            minimal::ImagePreperation::SortById(rings[k]);
            RingCloser::CloseRing(rings[k]);

            timer.Tick("Closed Center Ring");
            
            // TODO: Disabled for debugging. 
            //IterativeBundleAligner aligner;
    	    //aligner.Align(best, recorderGraph, imagesToTargets, 5, 0.5);

            timer.Tick("Bundle Adjustment Finished");

            BiMap<size_t, uint32_t> finalImagesToTargets;
            RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(recorderGraph, 2);

            vector<InputImageP> bestAlignment = halfGraph.SelectBestMatches(best, finalImagesToTargets, false);

            Log << "Post-Alignment, found " << bestAlignment.size() << "/" << halfGraph.Size() << "/" << best.size();

            timer.Tick("Found best matches for post alignment");

            if(fillMissingImages) {
                halfGraph.AddDummyImages(bestAlignment, finalImagesToTargets, Scalar(255, 0, 0), originalSize);
            }
           
            // #### Step 3: Convert to stereo, save results. 

            MonoStitcher stereoConverter;
            vector<vector<StereoImage>> stereoRings;

            minimal::ImagePreperation::SortById(bestAlignment);
            
            auto UnloadImage =
            	[] (const SelectionInfo &img) {
                   if (img.image->image.IsLoaded())
                   		img.image->image.Unload();
                };

            auto CreateStereo =
                [&] (const SelectionInfo &a, const SelectionInfo &b) {

         	    StereoImage stereo;
                SelectionEdge dummy;

                bool hasEdge = halfGraph.GetEdge(a.closestPoint, b.closestPoint, dummy);

           	    AssertWM(hasEdge, "Pair is correctly ordered");

                if(!hasEdge)
                    return;

                /*
                 *  Load the original image ( not the minified one )
                 */
                if (!a.image->image.IsLoaded())
                    a.image->image.Load();
                if (!b.image->image.IsLoaded())
                    b.image->image.Load();

                stereoConverter.CreateStereo(a, b, stereo);

                while(stereoRings.size() <= a.closestPoint.ringId) {
                    stereoRings.push_back(vector<StereoImage>());
                }
                stereoRings[a.closestPoint.ringId].push_back(stereo);

                /*
                 * Unload image to save memory
                 */
                if (a.image->image.IsLoaded())
                    a.image->image.Unload();
                if (b.image->image.IsLoaded())
                    b.image->image.Unload();

                leftStore.SaveRectifiedImage(stereo.A);
            	rightStore.SaveRectifiedImage(stereo.B);

                /*
                 * Unload image to save memory
                 */
                stereo.A->image.Unload();
                stereo.B->image.Unload();
            };

            auto FinishImage = [] (const SelectionInfo) { };

            RingProcessor<SelectionInfo> stereoRingBuffer(1, 1, UnloadImage, CreateStereo, FinishImage);

       	    int lastRingId = -1;
            for(auto img : bestAlignment) {
            	SelectionPoint target;
            	uint32_t pointId = 0;

            	Assert(finalImagesToTargets.GetValue(img->id, pointId));
            	Assert(halfGraph.GetPointById(pointId, target));

                // TODO: This is quite a hack. 
                double maxVFov = GetVerticalFov(img->intrinsics);
                target.vFov = maxVFov;
            	SelectionInfo info;
            	info.isValid = true;
            	info.closestPoint = target;
            	info.image = img;

            	if(lastRingId != -1 && lastRingId != (int)target.ringId) {
                    stereoRingBuffer.Flush();
            	}

            	stereoRingBuffer.Push(info);
            	lastRingId = target.ringId;
            }
            stereoRingBuffer.Flush();

            timer.Tick("Stereo Process");

        	/*
             * push the images to the stores , this will be used for the stitching
             */
            vector<InputImageP> rightImages;
            vector<InputImageP> leftImages;

            for(vector<StereoImage> rings : stereoRings) {
                for(StereoImage stereo : rings) {
                    leftImages.push_back(stereo.A);
                    rightImages.push_back(stereo.B);
                }

                vector<vector<InputImageP>> rightRings =
                    halfGraph.SplitIntoRings(rightImages);
                vector<vector<InputImageP>> leftRings =
                    halfGraph.SplitIntoRings(leftImages);

                leftStore.SaveStitcherInput(leftRings, gains );
                rightStore.SaveStitcherInput(rightRings, gains );
       		}

            timer.Tick("Save stitcher input");

        }
   };
}

#endif
