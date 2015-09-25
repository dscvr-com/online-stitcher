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

using namespace cv;
using namespace std;

#ifndef OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER

namespace optonaut {
	class RingwiseStreamAligner : public Aligner {
	private:
        typedef AsyncStream<ImageP, int> Worker; 
        RecorderGraph &graph;
		PairwiseCorrelator visual;
        vector<vector<ImageP>> rings;
        ImageP last;
        Mat compassDrift;
        shared_ptr<Worker> worker;
       
        double lasty;

        static constexpr double keyframeThreshold = M_PI / 64; 
        
        deque<ImageP> pasts;
        deque<double> sangles;

        const bool async;
	public:
		RingwiseStreamAligner(RecorderGraph &graph, const bool async = true) : 
            graph(graph), 
            visual(), 
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

        }

        vector<vector<ImageP>> SplitIntoRings(vector<ImageP> &imgs) const {
            
            vector<vector<ImageP>> rings(graph.GetRings().size());

            for(auto img : imgs) {
                int r = graph.FindAssociatedRing(img->originalExtrinsics);
                if(r == -1)
                    continue;
                rings[r].push_back(img);
            }

            return rings;
        }

        function<int(ImageP)> alignOp = [&] (ImageP next) -> int {

            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
            size_t target = graph.GetParentRing(ring);
                
            ImageP closest = NULL;
            double minDist = 100;

            Mat search = compassDrift * next->originalExtrinsics;

            for(size_t j = 0; j < rings[target].size(); j++) {
                double dist = abs(GetAngleOfRotation(search, rings[target][j]->adjustedExtrinsics));

                if(closest == NULL || dist < minDist) {
                    minDist = dist;
                    closest = rings[target][j];
                } 
            }

            
            if(closest != NULL) {
                CorrelationDiff corr = visual.Match(next, closest);
               
                if(corr.valid) { 
                    double dx = corr.offset.x;
                    double width = next->img.cols;
                    
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

		void Push(ImageP next) {
            
            last = next;

            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
           // cout << "Ring " << ring << endl;

            if(ring == -1)
                return; //No ring x(
           
            if(graph.HasChildRing(ring)) { 
                //Select keyframes based on selection points...
                if(rings[ring].size() < 1 || abs(GetAngleOfRotation(rings[ring].back()->originalExtrinsics, last->originalExtrinsics)) > keyframeThreshold) {
                    rings[ring].push_back(next);
                    cout << "Keyframe into ring " << ring << endl;
                }
            }

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

            static const int order = 5;

            if(sangles.size() > order) {
                sangles.pop_front();
            }
            if(pasts.size() > order) {
                pasts.pop_front();
            }
            if(sangles.size() == order) {
                double avg = Average(sangles, 1.0 / 5.0);
                CreateRotationY(avg, compassDrift);
                
                next->vtag = avg;
            }
        }

		Mat GetCurrentRotation() const {
			return compassDrift * last->originalExtrinsics;
		}

        void Finish() {
            if(async) {
                worker->Dispose();
                worker = NULL;
            }
        }

        void Postprocess(vector<ImageP> imgs) const {
            for(auto img : imgs) {
                //DrawBar(img->img, img->vtag);
            }
        }
	};
}

#endif
