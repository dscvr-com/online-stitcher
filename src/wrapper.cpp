#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "wrapper.hpp"
#include "core.hpp"
#include "support.hpp"
#include "streamAligner.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace optonaut {
namespace wrapper {

	StreamAligner state;
	Image *prev = NULL;
	bool debug = false;

	Image* AllocateImage(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, int id) {
		Mat inputExtrinsics = Mat(3, 3, CV_64F, extrinsics);
		Image *current = new Image();
		current->img = Mat(height, width, CV_8UC3);
		cvtColor(Mat(height, width, CV_8UC4, image), current->img, COLOR_RGBA2RGB);

		//IOS Base Conversion
		double baseV[] = {0, 1, 0,
						 -1, 0, 0, 
						 0, 0, 1};

	    Mat base(3,3, CV_64F, baseV);


		From3DoubleTo4Double(base * inputExtrinsics.inv() * base.inv(), current->extrinsics);
		current->intrinsics = Mat(3, 3, CV_64F, intrinsics);
		current->id = id;
		current->source = "dynamic";

		return current;
	}

	void Push(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, double newExtrinsics[], int id) {
		Image* current = AllocateImage(extrinsics, intrinsics, image, width, height, id);

		imwrite("dbg/pushed.jpg", current->img);

		state.Push(current);

		Mat e = state.GetCurrentRotation();
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				newExtrinsics[i * 4 + j] = e.at<double>(i, j);

		//Only safe because we know what goes on inside state. 
		if(prev != NULL && !debug) {
			delete prev;
		}

		prev = current;
	}

	void Debug() {
		debug = true;
	}

	void Free() {
		if(prev != NULL && !debug) {
			delete prev;
			prev = NULL;
		}
	}

	Image* GetLastImage() {
		return prev;
	}
}
}