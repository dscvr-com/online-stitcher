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
                cout << "[minifyImages] " << counter << " source: " << source << endl;

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

            vector<vector<InputImageP>> rings = recorderGraph.SplitIntoRings(miniImages);
            size_t k = rings.size() / 2;
            

            minimal::ImagePreperation::SortById(rings[k]);
            RingCloser::CloseRing(rings[k]);


   				  IterativeBundleAligner aligner;
    				aligner.Align(best, recorderGraph, imagesToTargets, 5, 0.5);

            
            int current_minified_height = miniImages[0]->image.cols;
            cout << "current_minified_height : " << current_minified_height << endl;
            // preLoad
            static int count = 0;
            auto loadFullImage = 
            	[] (const SelectionInfo &img) {
                   // unload the last minified image
                   // this is a cheat. need to find the bug
                   std::string str2 ("debug");
                   std::string str3 ("post");
                   std::size_t found = img.image->image.source.find(str2);
                   if (found!=std::string::npos) {
                        std::cout << "found 'debug' at: " << found << '\n';
                        img.image->image.source.replace(found,5,str3); 
                        cout << "image source replace " << img.image->image.source << endl;
                   }



                   if (img.image->image.IsLoaded())
                   		img.image->image.Unload();          
                   // load the full image ( data from the source att)
                   img.image->image.Load();          
                   count++;
                   cout << "image source " << img.image->image.source << endl;
                   cout << "image size " << img.image->image.size() << "  " << count << endl;
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

                if ( current_minified_height == a.image->image.cols ) 
                    a.image->image.Load();

                if ( current_minified_height == b.image->image.cols ) 
                    b.image->image.Load();


                cout << "current_minified_height " << current_minified_height << endl;
                cout << "a.image->image.cols " << a.image->image.cols << endl;
                cout << "b.image->image.cols " << b.image->image.cols << endl;
                cout << "img size a" << a.image->image.size() << endl;
                cout << "img size b" << b.image->image.size() << endl;
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
