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
#include "../recorder/stereoSink.hpp"
#include "../recorder/imageSink.hpp"
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

        CheckpointStore &imageStore;
        CheckpointStore &leftStore;
        CheckpointStore &rightStore;
        RecorderGraphGenerator generator;
    public:

        GlobalAlignment(CheckpointStore &imageStore, CheckpointStore &leftStore,
                        CheckpointStore &rightStore ) :
            imageStore(imageStore), leftStore(leftStore), rightStore(rightStore), generator() {
        }
 
        void minifyImages ( vector<InputImageP> &images, int downsample = 2) {
           AssertGT(downsample, 0);

           int  counter = 0;
           for(auto img : images) {            
                bool loaded = false;        
                std::string source = img->image.source;
                if(!img->image.IsLoaded()) {
                    loaded = true;              
                    img->image.Load();          
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
        

        void Finish() {
            
            vector<std::vector<InputImageP>> loadedImages;
            vector<std::vector<InputImageP>> fullImages;
            BiMap<size_t, uint32_t> imagesToTargets, d;
            map<size_t, double> gains;
            MonoStitcher stereoConverter;
            vector<vector<StereoImage>> stereoRings;
            imageStore.LoadStitcherInput(loadedImages, gains);

            cv::Size originalSize = fun::flat(loadedImages)[0]->image.size();
            vector <InputImageP> miniImages = fun::flat(loadedImages);
            int downsample = 3;
            // minify the images
            minifyImages(miniImages, downsample);
            Mat intrinsics;
            intrinsics = miniImages[0]->intrinsics;
            int graphConfiguration = 0;

            /*
             * check the number of rings and based use the correct graph configuration
             */
            if (loadedImages.size() == 1) {
               graphConfiguration =  RecorderGraph::ModeCenter;
            } else if (loadedImages.size() == 3)  {
               graphConfiguration =  RecorderGraph::ModeTruncated;
            }
  
            RecorderGraph recorderGraph = generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityNormal, 0, 8);
            
            vector<InputImageP> best = recorderGraph.SelectBestMatches(miniImages, imagesToTargets, false);

            vector<vector<InputImageP>> rings = recorderGraph.SplitIntoRings(miniImages);
            size_t k = rings.size() / 2;
            

            minimal::ImagePreperation::SortById(rings[k]);
            RingCloser::CloseRing(rings[k]);


            IterativeBundleAligner aligner;
    	    aligner.Align(best, recorderGraph, imagesToTargets, 5, 0.5);

            
            int current_minified_height = miniImages[0]->image.cols;
            // preLoad
            static int count = 0;
            auto loadFullImage = 
            	[] (const SelectionInfo &img) {
                   // unload the last minified image
                   if (img.image->image.IsLoaded())
                   		img.image->image.Unload();          
                };


            BiMap<size_t, uint32_t> finalImagesToTargets;
            RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(recorderGraph, 2);

            vector<InputImageP> bestAlignment = halfGraph.SelectBestMatches(best, finalImagesToTargets, false);

            if(debug) {
                halfGraph.AddDummyImages(bestAlignment, finalImagesToTargets, Scalar(255, 0, 0), originalSize);
            }

            sort(bestAlignment.begin(), bestAlignment.end(), [&] (auto a, auto b) {
                uint32_t aId, bId;
                Assert(finalImagesToTargets.GetValue(a->id, aId));
                Assert(finalImagesToTargets.GetValue(b->id, bId));

                return aId < bId;
            });

            auto ForwardToStereoProcess = 
                [&] (const SelectionInfo &a, const SelectionInfo &b) {
    
         	    StereoImage stereo;
                SelectionEdge dummy;        
            
           	    AssertM(halfGraph.GetEdge(a.closestPoint, b.closestPoint, dummy), "Pair is correctly ordered");

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

                count++;
                if(count % 100 == 0) {
                    cout << "Warning: Input Queue overflow: " <<  count << endl;
                    /*
                     *  Pause for 1 second every 100 images to minimize memory outage
                     */
                    this_thread::sleep_for(chrono::seconds(1));
                }
            };

            auto FinishImage = [] (const SelectionInfo) { };

            RingProcessor<SelectionInfo> stereoRingBuffer(1, 1, loadFullImage, ForwardToStereoProcess, FinishImage);
           
            
       	    int lastRingId = -1;
            for(auto img : bestAlignment) {
            	SelectionPoint target;
            	//Reassign points
            	uint32_t pointId = 0;
            	Assert(finalImagesToTargets.GetValue(img->id, pointId));
            	Assert(halfGraph.GetPointById(pointId, target));

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


        }
   };    
}

#endif
