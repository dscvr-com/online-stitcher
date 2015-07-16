
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


#ifndef OPTONAUT_WRAPPER_HEADER
#define OPTONAUT_WRAPPER_HEADER

namespace optonaut {
	void Push(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, double newExtrinsics[], int id);
}

#endif