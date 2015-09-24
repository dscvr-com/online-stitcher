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

using namespace cv;
using namespace std;

#ifndef OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER

namespace optonaut {
	class RingwiseStreamAligner : public Aligner {
	private:
        RecorderGraph &graph;
		PairwiseCorrelator visual;
        vector<vector<ImageP>> rings;
        ImageP last;
        Mat compassDrift;
       
        double lasty;
        double lastx;

        static constexpr double keyframeThreshold = M_PI / 64; 
        
        deque<ImageP> pasts;
        deque<double> sangles;
	public:
		RingwiseStreamAligner(RecorderGraph &graph) : 
            graph(graph), 
            visual(), 
            rings(graph.GetRings().size()), 
            compassDrift(Mat::eye(4, 4, CV_64F)), 
            lasty(0),
            lastx(0) 
        { }

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

		void Push(ImageP next) {
            
            last = next;

            int ring = graph.FindAssociatedRing(next->originalExtrinsics);
           // cout << "Ring " << ring << endl;

            if(ring == -1)
                return; //No ring x(
           
            if(graph.HasChildRing(ring)) { 
                assert(ring == 2);
                if(rings[ring].size() < 1 || abs(GetAngleOfRotation(rings[ring].back()->originalExtrinsics, last->originalExtrinsics)) > keyframeThreshold) {
                    rings[ring].push_back(next);
                    cout << "Keyframe into ring " << ring << endl;
                }
            }
            size_t target = graph.GetParentRing(ring);

           // cout << "Target " << target << endl;
            
            ImageP closest = NULL;
            double minDist = 100;

            if((int)target != ring) {
                for(size_t j = 0; j < rings[target].size(); j++) {
                    double dist = abs(GetAngleOfRotation(compassDrift * next->originalExtrinsics, rings[target][j]->adjustedExtrinsics));

                    if(closest == NULL || dist < minDist) {
                        minDist = dist;
                        closest = rings[target][j];
                    } 
                }
            }

            //cout << "Closest " << closest << endl;
            
            if(closest != NULL /*&& next->id % 10 == 0*/) {
                //cout << "Correlating " << next->id << " and " << closest->id << endl;
                CorrelationDiff corr = visual.Match(next, closest);
               
                if(corr.valid) { 
                    //Extract angles
                    double dx = corr.offset.x;
                    double dy = corr.offset.y;
                    double width = next->img.cols;
                    double height = next->img.rows;
                    
                    //cout << "dx: " << dx << ", width: " << width << endl; 

                    double hy = next->intrinsics.at<double>(1, 1) / (next->intrinsics.at<double>(1, 2) * 2); 
                    double hx = next->intrinsics.at<double>(0, 0) / (next->intrinsics.at<double>(0, 2) * 2); 

                    assert(dx <= width);
                    assert(dy <= height);

                    double angleY = asin((dx / width) / hx);
                    double angleX = asin((dy / height) / hy);

                    Mat rveca(3, 1, CV_64F);
                    Mat rvecb(3, 1, CV_64F);
                    Mat ror(4, 4, CV_64F);
                    ExtractRotationVector(closest->adjustedExtrinsics, rveca);
                    ExtractRotationVector(next->originalExtrinsics, rvecb);

                    angleX += GetDistanceY(next->originalExtrinsics, closest->adjustedExtrinsics);

                    if(angleX < -M_PI)
                        angleX = M_PI * 2 + angleX;
                    
                    if(angleY < -M_PI)
                        angleY = M_PI * 2 + angleY;

                    Mat rotY(4, 4, CV_64F);

                    //End extract angles
                    //cout << "Pushing for correspondance x: " << angleX << ", y: " << angleY << endl;

                    lasty = angleY;
                    lastx = angleX;
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
                
                //Timelapse here.
                //pasts.front()->adjustedExtrinsics = compassDrift * pasts.front()->originalExtrinsics;
                //pasts.front()->vtag = avg;
                next->vtag = avg;
            }
        }

		Mat GetCurrentRotation() const {
			return compassDrift * last->originalExtrinsics;
		}

        void Finish() {

        }

        void Postprocess(vector<ImageP> imgs) const {
            for(auto img : imgs) {
                //DrawBar(img->img, img->vtag);
            }
        }
	};
}

#endif
