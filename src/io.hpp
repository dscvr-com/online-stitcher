#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include <opencv2/opencv.hpp>
#include "core.hpp"

#ifndef OPTONAUT_IO_HEADER
#define OPTONAUT_IO_HEADER

namespace optonaut {

	bool StringEndsWith(const std::string& a, const std::string& b);
	void MatrixFromXml(tinyxml2::XMLElement* node, cv::Mat &out);

	Image* ImageFromFile(std::string path);
}

#endif