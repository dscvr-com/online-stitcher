#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "aligner.hpp"
#include "pairwiseVisualAligner.hpp"
#include "stat.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_RINGWISE_STREAM_ALIGNMENT_HEADER

namespace optonaut {
	class RingwiseStreamAligner : public Aligner {
	private:
		PairwiseVisualAligner visual;
        vector<vector<ImageP>> rings;
        ImageP last;
        Mat compassDrift;
        deque<double> angles;
	public:
		RingwiseStreamAligner() : visual(PairwiseVisualAligner::ModeECCAffine), compassDrift(Mat::eye(4, 4, CV_64F)) { }

        bool NeedsImageData() {
            return true;
        }

        void Dispose() {

        }

		void Push(ImageP next) {
            visual.FindKeyPoints(next);
            //TODO - adjust incoming with memorized compass drift. 
            next->originalExtrinsics = /*compassDrift **/ next->originalExtrinsics;
            
            size_t r = 0;

            for(auto ring : rings) {
                if(abs(GetDistanceY(ring[0]->adjustedExtrinsics, next->originalExtrinsics)) < M_PI / 8) {
                    break;
                }
                r++;
            } 

            cout << "Ring " << r << endl;

            if(r >= rings.size()) {
                rings.push_back(vector<ImageP>());
            }
            rings[r].push_back(next);

            ImageP closest = NULL;
            double minDist = 100;

            Mat nextRVec(3, 1, CV_64F);
            ExtractRotationVector(next->originalExtrinsics, nextRVec);

            if(0 != r) {
                for(size_t j = 0; j < rings[0].size(); j++) {
                    //TODO: Fix the metric. 
                    Mat imgRVec(3, 1, CV_64F);
                    ExtractRotationVector(rings[0][j]->adjustedExtrinsics,imgRVec);

                    double dist = abs(imgRVec.at<double>(1) - nextRVec.at<double>(1));

                    if(closest == NULL || dist < minDist) {
                        minDist = dist;
                        closest = rings[0][j];
                        //if(dist < M_PI / 4) {
                        //    MatchInfo* corr = visual.FindCorrespondence(next, closest);
                        //    cout << "Found " << corr->rotation << endl;
                        //}
                    } 
                }
            }
            
            if(closest != NULL) {
                MatchInfo* corr = visual.FindCorrespondence(next, closest);
               
                if(corr->valid) { 
                    double dx = corr->homography.at<double>(0, 2);
                    double width = next->img.cols;
                    
                    cout << "dx: " << dx << ", width: " << width << endl; 

                    double h = next->intrinsics.at<double>(1, 1) / (next->intrinsics.at<double>(1, 2) * 2); 
                    double angle = asin((dx / width) / h);

                    Mat rveca(3, 1, CV_64F);
                    Mat rvecb(3, 1, CV_64F);
                    Mat ror(4, 4, CV_64F);
                    ExtractRotationVector(closest->adjustedExtrinsics, rveca);
                    ExtractRotationVector(next->originalExtrinsics, rvecb);

                    angle = rveca.at<double>(1) - rvecb.at<double>(1) - angle;

                    Mat rotY(4, 4, CV_64F);

                    angles.push_back(angle);

                    cout << "Adjusting angle: " << angle << endl;
                    if(angles.size() >= 50) {
                        if(angles.size() > 50) {
                            angles.pop_front(); 
                        }
                        
                        angle = Average(angles, 1.0 / 5.0);
                        cout << "Adjusting angle: " << angle << endl;

                        CreateRotationY(angle, rotY); 

                        compassDrift = rotY;
                    }
                }
            }

            last = next;


        }

		Mat GetCurrentRotation() const {
			return compassDrift * last->originalExtrinsics;
		}
	};
}

#endif
