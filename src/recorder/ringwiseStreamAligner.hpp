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
        double cury;

        deque<InputImageP> pasts;
        deque<double> sangles;

        const bool async;
        static const bool debug = false;
	public:
		RingwiseStreamAligner(RecorderGraph &graph, ExposureCompensator&, const bool async = true) :
            graph(graph), 
            //visual(exposure),
            rings(graph.GetRings().size()), 
            compassDrift(Mat::eye(4, 4, CV_64F)), 
            lasty(0),
            cury(0),
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
            STimer aTime;
            InputImageP closest = GetClosestKeyframe(next->adjustedExtrinsics);
            //aTime.Tick("Keyframe found");
            if(closest != NULL) {
                //Todo: Bias intensity is probably dependent on image size. 
                CorrelationDiff corr = visual.Match(next, closest);
                //aTime.Tick("Aligned");

                double angleY = corr.horizontalAngularOffset;
                
                while(angleY < -M_PI)
                    angleY = M_PI * 2 + angleY;
                
                //Check variance. Value of 10^6 guessed via chart observation. 
                if(corr.valid && corr.variance > 1000) { 
                    lasty = angleY;
                } else {
                    lasty = cury;
                }
                //cout << "AngularDiffBias: " << lasty << endl;
                //cout << "AbsDiffBias: " << corr.offset.x << endl;
                //cout << "Variance: " << corr.variance << endl;
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
            
            int ring = graph.FindAssociatedRing(next->adjustedExtrinsics);
            // cout << "Ring " << ring << endl;
            if(ring == -1)
                return;
            
            size_t target = graph.GetParentRing(ring);
            
            // cout << "Target " << target << endl;
            if((int)target == ring)
                return;
            
            static const int order = 30;
            
            if(sangles.size() == order) {
                cury = Mean(sangles, 1.0 / 3.0);
                CreateRotationY(cury, compassDrift);
                cout << "FilteredBias: " << cury << endl;
            }

            if(async) {
                if(worker->Finished()) {
                    if(!next->IsLoaded()) {
                        //Pre-load the image. 
                        next->LoadFromDataRef();
                    }
                    worker->Push(next);
                }
            } else {
                if(!next->IsLoaded()) {
                    //Pre-load the image.
                    next->LoadFromDataRef();
                }
                alignOp(next);
            }
                
            sangles.push_back(lasty);
 
            if(sangles.size() > order) {
               sangles.pop_front();
            }
        }

		Mat GetCurrentBias() const {
			return compassDrift;
        }
        
        double GetCurrentVtag() const {
            return cury;
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
