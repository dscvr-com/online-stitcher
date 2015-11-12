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

        deque<InputImageP> pasts;
        deque<double> sangles;

        const bool async;
	public:
		RingwiseStreamAligner(RecorderGraph &graph, ExposureCompensator &exposure, const bool async = true) :
            graph(graph), 
            visual(exposure),
            rings(graph.GetRings().size()), 
            compassDrift(Mat::eye(4, 4, CV_64F)), 
            lasty(0),
            async(async)
        { 
            worker = shared_ptr<Worker>(new Worker(alignOp));
        }

        bool NeedsImageData() {
            return false; //Will always load ourselves. 
        }

        void Dispose() {
            if(async) {
                if(worker->IsRunning()) {
                    worker->Dispose();
                }
                worker = NULL;
            }
            
            rings.clear();
        }
        
        vector<vector<InputImageP>> SplitIntoRings(vector<InputImageP> &imgs) const {
            return graph.SplitIntoRings(imgs);
        }

        vector<KeyframeInfo> GetClosestKeyframes(const Mat &search, size_t count) const {

            int ring = graph.FindAssociatedRing(search);
            
            
            if(ring == -1)
                return { };

            int target = graph.GetParentRing(ring);
            
            if(target == ring)
                return { };
            
            assert(target != -1);
                
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

        function<int(InputImageP)> alignOp = [&] (InputImageP next) -> int {
            InputImageP closest = GetClosestKeyframe(next->adjustedExtrinsics);

            if(closest != NULL) {
                //Todo: Bias intensity is probably dependent on image size. 
                CorrelationDiff corr = visual.Match(next, closest);
               
                if(corr.valid) { 
                    double angleY = corr.horizontalAngularOffset;
                    
                    while(angleY < -M_PI)
                        angleY = M_PI * 2 + angleY;
                    
                    lasty = angleY;
                }
            }

            return lasty;
        };

        void AddKeyframe(InputImageP next) {
            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
            
            if(ring == -1)
                return; //No ring x(
           
            if(graph.HasChildRing(ring)) { 
                rings[ring].push_back(next);
            
                cout << "Adding keyframe " << next->id << " to ring: " << ring << " count: " << rings[ring].size() << endl;
            }


        }

		void Push(InputImageP next) {
            last = next;
            next->adjustedExtrinsics = compassDrift * next->originalExtrinsics;

            int ring = graph.FindAssociatedRing(next->adjustedExtrinsics);
            // cout << "Ring " << ring << endl;
            if(ring == -1)
                return;

            size_t target = graph.GetParentRing(ring);

            // cout << "Target " << target << endl;

            if((int)target != ring) {
                if(async) {
                    if(worker->Finished()) {
                        if(!next->IsLoaded()) {
                            //Pre-load the image. 
                            next->LoadFromDataRef();
                        }
cout << "RSA worker push " << endl;
                        worker->Push(next);
                    }
                } else {
                    if(!next->IsLoaded()) {
                        //Pre-load the image. 
                        next->LoadFromDataRef();
                    }
                    alignOp(next);
                }
            }

            double avg = 0;
            
            static const int order = 5;
            
            if(sangles.size() == order) {
                avg = Average(sangles, 1.0 / 5.0);
                CreateRotationY(avg, compassDrift);
                
                next->vtag = avg;
            }

            sangles.push_back(lasty + avg);
            pasts.push_back(next); 

            if(sangles.size() > order) {
                sangles.pop_front();
            }
            if(pasts.size() > order) {
                pasts.pop_front();
            }
        }

		Mat GetCurrentRotation() const {
			return compassDrift * last->originalExtrinsics;
		}

        void Finish() {
            if(async) {
                worker->Dispose();
            }
            rings.clear();
        }

        void Postprocess(vector<InputImageP> imgs) const {
            for(auto img : imgs) {
                DrawBar(img->image.data, img->vtag);
            }
        }
	};
}

#endif
