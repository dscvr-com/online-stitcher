#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "wrapper.hpp"
#include "image.hpp"
#include "support.hpp"
#include "streamAligner.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace optonaut {
namespace wrapper {

	size_t alignmentOrder = 3;
	StreamAligner aligner(alignmentOrder);
	ImageP prev = NULL;
	bool debug = false;
	deque<ImageP> images;
	string debugDir;


	//IOS Base Conversion
	//If baseV != baseV^-1, add inversion below. 
	double baseV[] = {0, 1, 0, 0,
					 1, 0, 0, 0,
					 0, 0, 1, 0, 
					 0, 0, 0, 1};

    Mat iosBase(4, 4, CV_64F, baseV);
       

    ImageP AllocateImage(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, int id) {
		Mat inputExtrinsics = Mat(4, 4, CV_64F, extrinsics).clone();

		ImageP current(new Image());
		images.push_back(current);

		current->img = Mat(height, width, CV_8UC3);
		cvtColor(Mat(height, width, CV_8UC4, image), current->img, COLOR_RGBA2RGB);

		current->extrinsics = iosBase * inputExtrinsics.inv() * iosBase;
		current->intrinsics = Mat(3, 3, CV_64F, intrinsics).clone();

		current->id = id;
		current->source = "dynamic";

		return current;
	}

    bool Push(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, double newExtrinsics[], int id) {
        
        ImageP current = AllocateImage(extrinsics, intrinsics, image, width, height, id);

		aligner.Push(current);

		Mat e = iosBase * (aligner.GetCurrentRotation() * aligner.GetZero()) * iosBase;
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				newExtrinsics[i * 4 + j] = e.at<double>(i, j);

		prev = current;

		return true;
	}

	void EnableDebug(string dbgDir) {
		debug = true;
		debugDir = dbgDir;
	}

	void Free() {
		//TODO: Flush aligner history
	}

	ImageP GetLastImage() {
		return prev;
	}
}
}