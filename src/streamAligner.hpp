#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "aligner.hpp"
#include "visualAligner.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_STREAM_ALIGNMENT_HEADER

namespace optonaut {
	class StreamAligner : public Aligner {
	private:
		VisualAligner visual;
		deque<ImageP> previous;
		deque<Mat> rPrevious;
		deque<Mat> rSensorPrevious;
		size_t order;
	public:
		StreamAligner(size_t order = 3) : visual(), order(order) { }

        bool NeedsImageData() {
            return true;
        }

        void Dispose() {

        }

		void Push(ImageP next) {

            cout << "Stream aligner receiving: " << next->id << endl;

			visual.FindKeyPoints(next);

			if(previous.size() == 0) {
				//First!
				rPrevious.push_back(next->originalExtrinsics);
                //cout << rPrevious.back() << endl;
			} else {

				Mat sensorRVec(3, 1, CV_64F);
				Mat visualRVec(3, 1, CV_64F);

				//2nd. Do approximation/decision 
				//cout << "Finding homographies " << endl;

				int visualAnchor = -1;
				Mat visualDiff = Mat::eye(4, 4, CV_64F);

				for(size_t i = 0; i < previous.size(); i++) {
	        		MatchInfo* hom = visual.FindHomography(previous[i], next);
	        		if(hom->valid) { //Basic validity

		        		Mat rotation(4, 4, CV_64F);
		        		From3DoubleTo4Double(hom->rotation, rotation);
		        		rotation = rotation.inv();

                        if(visualAnchor == -1 || 
                            GetAngleOfRotation(rPrevious[0].t() * rPrevious[visualAnchor] * visualDiff) > 
                            GetAngleOfRotation(rPrevious[0].t() * rPrevious[i] * rotation)) {
                            visualAnchor = (int)i;
                            visualDiff = rotation;
                        }
		        	}
		        	delete hom;
				}

				if(visualAnchor != -1) {
					//cout << "Picking anchor between " << next->id << " <-> " << (next->id - (int)previous.size() + visualAnchor) << endl;
				} else {
					//cout << "Could not find visual anchor." << endl;
				}

        		//cout << "Image " << next->id << endl;
        		Mat sensorDiff;

        		if(visualAnchor != -1) {
	        		//cout << "Visual diff " << visualRVec.t() << endl;
	        		sensorDiff = rSensorPrevious[visualAnchor].t() * (next->originalExtrinsics);
				} else {
	        		sensorDiff = rSensorPrevious.back().t() * (next->originalExtrinsics);
				}

        		//cout << "Sensor diff " << sensorRVec.t() << endl;

        		//rPrevious = rPrevious * sensorDiff;

        		//Todo: Might replace this by a kalman-filtering model, if
        		//we understand the error modelling better. 

				//Mat offset(4, 4, CV_64F);
	        	//CreateRotationY(0.005, offset);
	        	//CreateRotationY(0.000, offset);

        		if(visualAnchor == -1 || GetAngleOfRotation(visualDiff * sensorDiff.inv()) > 0.5) {
                    //Fallback to sensor. 
	        		rPrevious.push_back(GetCurrentRotation() * sensorDiff);
	        		cout << "Sensor" << endl;
	        	} else {
                    if(GetAngleOfRotation(visualDiff * sensorDiff.inv()) > 0.1) {
                        //Filter out huge errors!
	        		    cout << "Sensor" << endl;
	        		    rPrevious.push_back(rPrevious[visualAnchor] * sensorDiff);
                    }
                    else 
                    {
                        //Combine sensor and visual data - sensor does great job for up/down/angle 
                        Mat vv, sv;

                        ExtractRotationVector(visualDiff, vv);
                        ExtractRotationVector(sensorDiff, sv);

                        Mat cv = sensorDiff;
                        cv.at<double>(1) = visualDiff.at<double>(1);

                        Mat rz, ry, rx;

                        CreateRotationX(cv.at<double>(0), rx);                        
                        CreateRotationY(cv.at<double>(1), ry);
                        CreateRotationZ(cv.at<double>(2), rz);                        
	        		
	        		    rPrevious.push_back(rPrevious[visualAnchor] * rx * ry * rz);
	        		    cout << "Visual" << endl;
                    }

	        	} 
			}

			previous.push_back(next);
			rSensorPrevious.push_back(next->originalExtrinsics.clone());

			if(previous.size() > order) {
				previous.pop_front();
				rSensorPrevious.pop_front();
				rPrevious.pop_front();
			}

		}

		Mat GetCurrentRotation() const {
			return rPrevious.back();
		}
	};
}

#endif
