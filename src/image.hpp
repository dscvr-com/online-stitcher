#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
	struct Image {
		cv::Mat img;
		cv::Mat extrinsics;
		cv::Mat intrinsics; 
		int id;
		std::string source;

		std::vector<cv::KeyPoint> features;
		cv::Mat descriptors;
	};

	typedef std::shared_ptr<Image> ImageP;
}


#endif