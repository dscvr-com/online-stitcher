#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "../imgproc/pairwiseCorrelator.hpp"
#include "../math/stat.hpp"
#include "../math/support.hpp"
#include "../common/image.hpp"
#include "../common/asyncStreamWrapper.hpp"

#include "aligner.hpp"
#include "recorderGraph.hpp"
#include "exposureCompensator.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER

namespace optonaut {
    /*
     * Class that alignes images based on images with a known position.
     *
     * Usually outer rings are compared towards a center baseline. 
     */
	class RingwiseStreamAligner : public Aligner {
	private:
        typedef AsyncStream<InputImageP, int> Worker; 
        RecorderGraph &graph;
		PairwiseCorrelator visual;
        vector<vector<InputImageP>> rings;
        InputImageP last;
        Mat compassDrift;
        shared_ptr<Worker> worker;
       
        double lasty;
        double cury;

        deque<InputImageP> pasts;
        deque<double> sangles;

        const bool async;
        static const bool debug = false;
	public:
        /*
         * Creates a new instance of this class, based on the given recorder graph. 
         */
		RingwiseStreamAligner(RecorderGraph &graph, const bool async = true) :
            graph(graph), 
            rings(graph.GetRings().size()), 
            compassDrift(Mat::eye(4, 4, CV_64F)), 
            lasty(0),
            cury(0),
            async(async)
        {
            AssertFalseInProduction(debug);
            worker = shared_ptr<Worker>(new Worker([this] (const InputImageP &x) { return AlignOperation(x); }));
        }

        bool NeedsImageData() {
            return false; //Will always load ourselves. 
        }

        /*
         * Stops operation and clears all resources. 
         */
        void Dispose() {
            if(async) {
                if(worker->IsRunning()) {
                    worker->Dispose();
                }
                worker = NULL;
            }
            
            rings.clear();
        }
       
        /*
         * Convenience method that splits a set of images into a set of rings. 
         */ 
        vector<vector<InputImageP>> SplitIntoRings(vector<InputImageP> &imgs) const {
            return graph.SplitIntoRings(imgs);
        }

        /*
         * Finds the closest known keyframes for a given position. 
         */
        vector<KeyframeInfo> GetClosestKeyframes(const Mat &search, size_t count) const {

            int ring = graph.FindAssociatedRing(search);
            
            if(ring == -1)
                return { };

            int target = graph.GetParentRing(ring);
            
            if(target == ring)
                return { };
            
            Assert(target != -1);
                
            InputImageP closest = NULL;
            
            vector<KeyframeInfo> copiedKeyframes(rings[target].size());
            
            count = count > copiedKeyframes.size() ? copiedKeyframes.size() : count;
            
            for(size_t i = 0; i < copiedKeyframes.size(); i++) {
                copiedKeyframes[i].dist = abs(GetAngleOfRotation(search, rings[target][i]->adjustedExtrinsics));
                copiedKeyframes[i].keyframe = rings[target][i];
            }
            
            vector<KeyframeInfo> keyframes(count);
            
            std::sort(copiedKeyframes.begin(), copiedKeyframes.end(),
                      [search](KeyframeInfo l, KeyframeInfo r)
                      {
                          return l.dist < r.dist;
                      });

            std::copy(copiedKeyframes.begin(), copiedKeyframes.begin() + count, keyframes.begin());

            return keyframes;
        }

        /*
         * Correlation operation that is executed for each image. 
         */
        int AlignOperation(const InputImageP &next) {
            STimer aTime;
            // Find the closest keyframe
            InputImageP closest = GetClosestKeyframe(next->adjustedExtrinsics);
            if(closest != NULL) {
                
                // Correlate the image and its closest keyrame.
                CorrelationDiff corr = visual.Match(CloneAndDownsample(next), closest);
                
                //aTime.Tick("Aligned");
                
                double angleY = corr.angularOffset.y;
                
                // Check variance. Value of 10^6 guessed via chart observation. 
                if(corr.valid) {
                    //cout << "Correlating with angular offset " << angleY << ", variance " << corr.correlationCoefficient << endl;
                    
                    while(angleY < -M_PI)
                        angleY = M_PI * 2 + angleY;
                   
                    // Remember image.
                    lasty = angleY;
                } else {
                    //cout << "Rejecting Corrleation with angular offset " << angleY << ", variance " << corr.correlationCoefficient << endl;
                    lasty = cury;
                }
                //cout << "AngularDiffBias: " << lasty << endl;
                //cout << "AbsDiffBias: " << corr.offset.x << endl;
                //cout << "Variance: " << corr.variance << endl;
            } else {
                cout << "Warning: No keyframe." << endl;
            }

            return lasty;
        };
               
        /*
         * Registers a new keyframe with this aligner. 
         */ 
        void AddKeyframe(InputImageP next) {
            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
            
            if(ring == -1)
                return; //No ring x(

            if(graph.HasChildRing(ring)) { 
                AssertGE(ring, 0);
                AssertGT((int)rings.size(), ring);
                rings[ring].push_back(next);
            
                //cout << "Adding keyframe " << next->id << " to ring: " << ring << " count: " << rings[ring].size() << endl;
            }
        }

        /*
         * Pushes a new image to this aligner. 
         * If the aligner is ready for new input, the image will be loaded, 
         * if necassary, and used to esimate sensor bias. 
         */
		void Push(InputImageP next) {
            last = next;
            
            int ring = graph.FindAssociatedRing(next->adjustedExtrinsics);
            // cout << "Ring " << ring << endl;
            if(ring == -1)
                return;
            
            size_t target = graph.GetParentRing(ring);
            
            // cout << "Target " << target << endl;
            if((int)target == ring)
                return;
            
            static const int order = 3;
            
            if(sangles.size() >= order / 2) {
                cury = Mean(sangles);
                CreateRotationY(cury, compassDrift);
                //cout << "FilteredBias: " << cury << endl;
            }

            if(async) {
                if(worker->Finished()) {
                    if(!next->IsLoaded()) {
                        //Pre-load the image. 
                        AssertM(false, "Tried to load an image from its data ref where it is potentially unsafe. Make sure that code runs on the AVFoundation's thread");
                        next->LoadFromDataRef();
                    }
                    worker->Push(next);
                }
            } else {
                if(!next->IsLoaded()) {
                    //Pre-load the image.
                    AssertM(false, "Tried to load an image from its data ref where it is potentially unsafe. Make sure that code runs on the AVFoundation's thread");
                    next->LoadFromDataRef();
                }
                AlignOperation(next);
            }
               
            sangles.push_back(lasty);
 
            if(sangles.size() > order) {
               sangles.pop_front();
            }
        }

		/*
         * Returns the current bias as 3x3 rotation matrix. 
         */
        Mat GetCurrentBias() const {
			return compassDrift;
        }
       
        /*
         * Returns the current angular offset around the vertical axis. 
         */ 
        double GetCurrentAngularOffset() const {
            return cury;
        }

        /*
         * Stops the worker thread. 
         */
        void Finish() {
            if(async) {
                worker->Dispose();
            }
        }

        /*
         * If debug enabled, saves all keyframes.  
         */
        void Postprocess(vector<InputImageP>) const {
            //for(auto img : imgs) {
            //    DrawBar(img->image.data, img->vtag);
            //}
            if(debug) {
                for(size_t i = 0; i < rings.size(); i++) {
                    SimpleSphereStitcher stitcher; 
                    auto res = stitcher.Stitch(rings[i]);
                    imwrite("dbg/keyframe_ring_" + ToString(i) + ".jpg", rings[i]);
                }
            }
        }
	};
}

#endif
