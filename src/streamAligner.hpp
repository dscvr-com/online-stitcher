#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "visualAligner.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_STREAM_ALIGNMENT_HEADER
#define OPTONAUT_STREAM_ALIGNMENT_HEADER

namespace optonaut {
	class StreamAligner {
	private:
		VisualAligner visual;
		deque<ImageP> previous;
		Mat rOrigin;
		deque<Mat> rPrevious;
		deque<Mat> rSensorPrevious;
		size_t order;
	public:
		StreamAligner(size_t order = 3) : visual(), rOrigin(4, 4, CV_64F), order(order) { }

		double Push(ImageP next) {

			visual.FindKeyPoints(next);

			if(previous.size() == 0) {
				//First!
				rOrigin = next->extrinsics;
				rPrevious.push_back(Mat::eye(4, 4, CV_64F));
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
		        		From3DoubleTo4Double(hom->rotations[0], rotation);
		        		rotation = rotation.inv();

	        			ExtractRotationVector(rotation, visualRVec);
						//cout << next->id << " <-> " << previous[i]->id<< " R: " << visualRVec.t() << " T: " << hom->translations[0].t() << endl;

		        		//Filter out stupid homomomomographies! 
		        		if(visualRVec.at<double>(2, 0) > 0.0 && //Don't allow "backward" rotation
		        			abs(visualRVec.at<double>(0, 0)) < 0.02 && //Don't allow rotation around other axis
		        			abs(visualRVec.at<double>(2, 0)) < 0.02) { 
		        			if(visualAnchor == -1 || 
		        				GetAngleOfRotation(rPrevious[0].inv() * rPrevious[visualAnchor] * visualDiff) > 
		        				GetAngleOfRotation(rPrevious[0].inv() * rPrevious[i] * rotation)) {
		        				visualAnchor = i;
		        				visualDiff = rotation;
		        			}
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
	        		ExtractRotationVector(visualDiff, visualRVec);
	        		//cout << "Visual diff " << visualRVec.t() << endl;
	        		sensorDiff = rSensorPrevious[visualAnchor].t() * (next->extrinsics);
	        		ExtractRotationVector(sensorDiff, sensorRVec);
				} else {
	        		sensorDiff = rSensorPrevious.back().t() * (next->extrinsics);
	        		ExtractRotationVector(sensorDiff, sensorRVec);
				}

        		//cout << "Sensor diff " << sensorRVec.t() << endl;

        		//rPrevious = rPrevious * sensorDiff;

        		//Todo: Might replace this by a kalman-filtering model, if
        		//we understand the error modelling better. 

				Mat offset(4, 4, CV_64F);
	        	CreateRotationY(0.005, offset);
	        	//CreateRotationY(0.000, offset);

        		if(visualAnchor == -1) {
	        		rPrevious.push_back(GetCurrentRotation() * sensorDiff * offset);
	        		//cout << "Sensor" << endl;
	        	} else if(GetAngleOfRotation(sensorDiff) > GetAngleOfRotation(visualDiff) * 2) {
	        		//If your sensor moved a lot more, discard!
	        		rPrevious.push_back(rPrevious[visualAnchor] * visualDiff * offset);
	        		//cout << "Visual" << endl;
	        	} else {

	        		//rPrevious.push_back(rPrevious[visualAnchor] * visualDiff);
	        		//cout << "Visual" << endl;

	        		//Use sensor - it's our best bet since everything except y rotation is very well measured
	        		//rPrevious = rPrevious * sensorDiff;
	        		//cout << "Combined" << endl;

	        		Mat mx(4, 4, CV_64F);
	        		Mat my(4, 4, CV_64F);
	        		Mat mz(4, 4, CV_64F);

	        		CreateRotationX(sensorRVec.at<double>(0, 0), mx);
	        		CreateRotationY(sensorRVec.at<double>(1, 0), my);
	        		CreateRotationZ(visualRVec.at<double>(2, 0), mz);

	        		rPrevious.push_back(rPrevious[visualAnchor] * (mx * my * mz) * offset);
	        		
	        	}
			}

			previous.push_back(next);
			rSensorPrevious.push_back(next->extrinsics.clone());

			if(previous.size() > order) {
				previous.pop_front();
				rSensorPrevious.pop_front();
				rPrevious.pop_front();
			}

			return 0;
		}

		const Mat &GetZero() {
			return rOrigin;
		}

		const Mat &GetCurrentRotation() {
			return rPrevious.back();
		}
	};
}

#endif
