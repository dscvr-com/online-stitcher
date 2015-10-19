#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "aligner.hpp"
#include "pairwiseCorrelator.hpp"
#include "stat.hpp"
#include "simpleSphereStitcher.hpp"
#include "recorderGraph.hpp"
#include "asyncStreamWrapper.hpp"
#include "exposureCompensator.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER


namespace optonaut {
    struct KeyframeInfo {
        InputImageP keyframe;
        double dist;
    };
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

        static constexpr double keyframeThreshold = M_PI / 64; 
        
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
            return true; 
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
            return SplitIntoRings(imgs, graph);
        }
        
        InputImageP GetClosestKeyframe(const Mat &search) {
            auto keyframes = GetClosestKeyframes(search, 1);
            if(keyframes.size() != 1) {
                return NULL;
            } else {
                return keyframes[0].keyframe;
            }
        }

        vector<KeyframeInfo> GetClosestKeyframes(const Mat &search, size_t count) {

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

        static vector<vector<InputImageP>> SplitIntoRings(vector<InputImageP> &imgs, RecorderGraph &graph) {
            
            vector<vector<InputImageP>> rings(graph.GetRings().size());

            for(auto img : imgs) {
                int r = graph.FindAssociatedRing(img->originalExtrinsics);
                if(r == -1)
                    continue;
                rings[r].push_back(img);
            }

            return rings;
        }

        function<int(InputImageP)> alignOp = [&] (InputImageP next) -> int {

            InputImageP closest = GetClosestKeyframe(next->originalExtrinsics);
            
            if(closest != NULL) {
                CorrelationDiff corr = visual.Match(next, closest);
               
                if(corr.valid) { 
                    double dx = corr.offset.x;
                    double width = next->image.cols;
                    
                    //cout << "dx: " << dx << ", width: " << width << endl; 

                    double hx = next->intrinsics.at<double>(0, 0) / (next->intrinsics.at<double>(0, 2) * 2); 

                    assert(dx <= width);

                    double angleY = asin((dx / width) / hx);

                    //Todo: Don't we have to include the diff between image/keyframe
                    //in this calculation. 
                    //(ej): No, we don't. The correlator includes the diff in 
                    //the projection calculation, the diff is exlusive 
                    //the previous compass drift. 
                    //angleY += GetDistanceY(search, closest->adjustedExtrinsics);
                    
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
            }

        }

		void Push(InputImageP next) {
            
            last = next;

            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
           // cout << "Ring " << ring << endl;
            if(ring == -1)
                return;

            size_t target = graph.GetParentRing(ring);

           // cout << "Target " << target << endl;

            if((int)target != ring) {
                if(async) {
                    worker->Push(next);
                } else {
                    alignOp(next);
                }
            }

            pasts.push_back(next); 
            sangles.push_back(lasty);

            static const int order = 100;

            if(sangles.size() > order) {
                sangles.pop_front();
            }
            if(pasts.size() > order) {
                pasts.pop_front();
            }
            if(sangles.size() == order) {
                double avg = Average(sangles, 1.0 / 5.0);
                //CreateRotationY(avg, compassDrift);
                
                next->vtag = avg;
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
                //DrawBar(img->img, img->vtag);
            }
        }
	};
}

#endif
