#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "core.hpp"
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
		Image* previous;
		Mat rOrigin;
		Mat rPrevious;

		Mat rSensorPrevious;
	public:
		StreamAligner() : visual(), previous(NULL), rOrigin(4, 4, CV_64F),
						  rPrevious(4, 4, CV_64F), rSensorPrevious(4, 4, CV_64F) { }

		double Push(Image* next) {

			visual.FindKeyPoints(next);

			if(previous == NULL) {
				//First!
				rOrigin = next->extrinsics;
				rPrevious = Mat::eye(4, 4, CV_64F);
			} else {

				Mat sensorRVec(3, 1, CV_64F);
				Mat visualRVec(3, 1, CV_64F);

				//2nd. Do approximation/decision 
        		MatchInfo* hom = visual.FindHomography(previous, next);

        		//cout << "Image " << next->id << endl;
				Mat visualDiff;
        		
        		if(hom->valid) {
	        		From3DoubleTo4Double(hom->rotations[0], visualDiff);
	        		ExtractRotationVector(visualDiff, visualRVec);
	        		//cout << "Visual diff " << visualRVec.t() << endl;
				}

        		Mat sensorDiff = (rSensorPrevious).t() * (next->extrinsics);
        		ExtractRotationVector(sensorDiff, sensorRVec);
        		//cout << "Sensor diff " << sensorRVec.t() << endl;

        		//rPrevious = rPrevious * sensorDiff;

        		//Todo: Might replace this by a kalman-filtering model, if
        		//we understand the error modelling better. 

        		if(!hom->valid || abs(visualRVec.at<double>(2, 0)) > 0.1) {
	        		//if the homoghraphy is invalid or the homography shows a big drift on the z-axis, discard.
	        		//z-axis drift should never happen, at least not on the middle ring. 
	        		rPrevious = rPrevious * sensorDiff;
	        		//cout << "Sensor" << endl;
	        	} else if(GetAngleOfRotation(sensorDiff) > GetAngleOfRotation(visualDiff) * 2) {
	        		//If your sensor moved a lot more, discard!
	        		rPrevious = rPrevious * visualDiff.inv();
	        		//cout << "Visual" << endl;
	        	} else {
	        		//Use sensor - it's our best bet since everything except y rotation is very well measured
	        		rPrevious = rPrevious * visualDiff.inv();
	        		//cout << "Sensor" << endl;
	        	}
			}

			previous = next;
			rSensorPrevious = previous->extrinsics.clone();

			return 0;
		}

		const Mat &GetZero() {
			return rOrigin;
		}

		const Mat &GetCurrentRotation() {
			return rPrevious;
		}
	};
}

#endif