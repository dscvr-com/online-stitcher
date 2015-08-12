
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "image.hpp"

#ifndef OPTONAUT_WRAPPER_HEADER
#define OPTONAUT_WRAPPER_HEADER

namespace optonaut {
namespace wrapper {
	void EnableDebug(std::string debugDir);
	bool Push(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, double newExtrinsics[], int id);
	ImageP GetLastImage();
	void Free();
    ImageP AllocateImage(double extrinsics[], double intrinsics[], unsigned char *image, int width, int height, int id);
}
}
#endif