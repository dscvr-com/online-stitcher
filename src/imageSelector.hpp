#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "core.hpp"
#include "support.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_IMAGE_SELECTOR_HEADER
#define OPTONAUT_IMAGE_SELECTOR_HEADER

namespace optonaut {

class ImageSelector {

private:
	vector<Mat> targets;
	Mat intrinsics;
	//Horizontal and Vertical overlap in procent. 
	const double hOverlap = 0.8;
	const double vOverlap = 0.3;

	//Tolerance, measured on sphere, for hits. 
	const double tolerance = M_PI / 32;
public:

	ImageSelector(const Mat &intrinsics) { 

		this->intrinsics = intrinsics;

		double hFov = GetHorizontalFov(intrinsics) * (1.0 - hOverlap);
		double vFov = GetVerticalFov(intrinsics) * (1.0 - vOverlap);

		int vCount = ceil(M_PI / vFov);
		int hCenterCount = ceil(2 * M_PI / hFov);
	
		vFov = M_PI / vCount;

		for(int i = 0; i < vCount; i++) {

			double vAngle = i * vFov + vFov / 2 - M_PI / 2;

			int hCount = hCenterCount * cos(vAngle);
			hFov = M_PI * 2 / hCount;

			for(int j = 0; j < hCount; j++) {
				Mat hRot;
				Mat vRot;


				CreateRotationY(j * hFov + hFov / 2, hRot);
				CreateRotationX(vAngle, vRot);

				targets.push_back(hRot * vRot);
			}
		} 
	}

	vector<Image*> GenerateDebugImages() {
		vector<Image*> images;

		for(auto t : targets) {

			Image *d = new Image();

			d->extrinsics = t;
			d->intrinsics = intrinsics;
			d->img = Mat::zeros(240, 320, CV_8UC3);
			d->id = 0;

			line(d->img, Point2f(0, 0), Point2f(320, 240), Scalar(0, 255, 0), 4);
			line(d->img, Point2f(320, 0), Point2f(0, 240), Scalar(0, 255, 0), 4);

			images.push_back(d);
		}

		return images;
	}

	bool FitsModel(Image* img) {
		auto i = targets.begin();
		Mat eInv = img->extrinsics.inv();

		while (i != targets.end()) {

		    if (GetAngleOfRotation(eInv * *i) < tolerance) {
		    	targets.erase(i);
		    	return true;
		    }
		}

		return false;
	}
};
}

#endif