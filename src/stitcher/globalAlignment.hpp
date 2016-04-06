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
#include "../recorder/recorderGraphGenerator.hpp"
#include "../recorder/iterativeBundleAligner.hpp"

#include <chrono>


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
  
                if(loaded) {   
                    img->image.Unload();            
                }
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
            cout << "Loading images " << endl;
            imageStore.LoadStitcherInput(loadedImages, gains);
            cout << "Done Loading images. minifying " << endl;
            vector <InputImageP> miniImages = fun::flat(loadedImages);
            int downsample = 3;
            // minify the images
            minifyImages(miniImages, downsample);
            cout << "Done minifying " << endl;

            Mat intrinsics;
            intrinsics = miniImages[0]->intrinsics;
            int ringsize = loadedImages.size();
            int graphConfiguration = 0;

            if (ringsize == 1) {
               graphConfiguration =  RecorderGraph::ModeCenter;
               cout << "ModeCenter" << endl;
            } else if (ringsize == 3)  {
               cout << "ModeTruncated" << endl;
               graphConfiguration =  RecorderGraph::ModeTruncated;
            }
  
            RecorderGraph recorderGraph = generator.Generate(intrinsics, graphConfiguration, RecorderGraph::DensityNormal, 0, 8);
            
            vector<InputImageP> best = recorderGraph.SelectBestMatches(miniImages, imagesToTargets, true);
            cout << "best size" << ToString(best.size()) << endl;
            cout << "miniimages size" << ToString(miniImages.size()) << endl;


   				  IterativeBundleAligner aligner;
    				aligner.Align(best, recorderGraph, imagesToTargets, 5, 0.5);

            // preLoad
            auto loadFullImage = 
            	[&] (const SelectionInfo &img) {
                   img.image->image.Load();          
            };

						// onProcess
            auto ForwardToStereoProcess = 
            	[&] (const SelectionInfo &a, const SelectionInfo &b) {
    
         				StereoImage stereo;
            		SelectionEdge dummy;        
            
           	 		if(!recorderGraph.GetEdge(a.closestPoint, b.closestPoint, dummy)) {
                	cout << "Skipping pair due to misalignment." << endl;
                	return;
           	 		}

            		stereoConverter.CreateStereo(a, b, stereo);

            		while(stereoRings.size() <= a.closestPoint.ringId) {
              	  stereoRings.push_back(vector<StereoImage>());
            		}
            		stereoRings[a.closestPoint.ringId].push_back(stereo);
            };

            auto FinishImage = [] (const SelectionInfo) { };


            RingProcessor<SelectionInfo> stereoRingBuffer(1, 1, loadFullImage, ForwardToStereoProcess, FinishImage);
           
            cout << "fullGraph size" << recorderGraph.Size() << endl;

            BiMap<size_t, uint32_t> finalImagesToTargets;
            RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(recorderGraph, 2);

            vector<InputImageP> bestAlignment = halfGraph.SelectBestMatches(best, finalImagesToTargets, true);
            cout << "bestAlignment size" << bestAlignment.size() << endl;
            cout << "halfGraph size" << halfGraph.Size() << endl;
            
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

        		// push the images to the stores
            vector<InputImageP> rightImages;
            vector<InputImageP> leftImages;

        		for(vector<StereoImage> rings : stereoRings) {
        		    for(StereoImage stereo : rings) {

            	     leftStore.SaveRectifiedImage(stereo.A);
            	     rightStore.SaveRectifiedImage(stereo.B);

                   leftImages.push_back(stereo.A);
                   rightImages.push_back(stereo.B);
                               
                 	 stereo.A->image.Unload();
                 	 stereo.B->image.Unload();
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
