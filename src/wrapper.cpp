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

	StreamAligner aligner;
	Image *prev = NULL;
	bool debug = false;

    Image* AllocateImage(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, int id, std::string dir) {
		Mat inputExtrinsics = Mat(4, 4, CV_64F, extrinsics).clone();
//        Mat inputExtrinsics = Mat::eye(4, 4, CV_64F);
		Image *current = new Image();
		current->img = Mat(height, width, CV_8UC3);
		cvtColor(Mat(height, width, CV_8UC4, image), current->img, COLOR_RGBA2RGB);

		//IOS Base Conversion
		double baseV[] = {0, 1, 0, 0,
						 1, 0, 0, 0,
						 0, 0, 1, 0, 
						 0, 0, 0, 1};

	    Mat base(4, 4, CV_64F, baseV);
        
//        imwrite(dir + "/test.jpg", current->img);



		current->extrinsics = base * inputExtrinsics.inv() * base.inv();
		current->intrinsics = Mat(3, 3, CV_64F, intrinsics).clone();
//        current->intrinsics = Mat::eye(3, 3, CV_64F);
		current->id = id;
		current->source = "dynamic";

		return current;
	}

    bool Push(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, double newExtrinsics[], int id, std::string dir) {
		
//        FILE* d;
//        d = fopen((dir + "/raw.out").c_str(), "w");
//        fwrite(image, 1, sizeof(unsigned char) * 4 * width * height, d);
//        fclose(d);
        
        Image* current = AllocateImage(extrinsics, intrinsics, image, width, height, id, dir);

//		imwrite(dir + "/dbg-pushed.jpg", current->img);

		aligner.Push(current);

		Mat e = aligner.GetCurrentRotation();
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				newExtrinsics[i * 4 + j] = e.at<double>(i, j);

		//Only safe because we know what goes on inside the StreamAligner. 
		if(prev != NULL && !debug) {
//			delete prev;
		}

		prev = current;

		return true;
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