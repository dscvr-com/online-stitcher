#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

namespace optonaut {

	bool StringEndsWith(const std::string& a, const std::string& b);
	cv::Mat MatrixFromXml(tinyxml2::XMLElement* node);
	int ParseInt(const char* data);

	std::string ToString(int i);

	void ScaleIntrinsicsToImage(cv::Mat intrinsics, cv::Mat image, cv::Mat &scaled);

	double GetHorizontalFov(cv::Mat intrinsics);

	cv::Mat ExtractRotationVector(cv::Mat r);
	double GetAngleOfRotation(cv::Mat r);

	double GetDistanceByDimension(cv::Mat a, cv::Mat b, int dim);
	double GetDistanceX(cv::Mat a, cv::Mat b);
	double GetDistanceY(cv::Mat a, cv::Mat b);
	double GetDistanceZ(cv::Mat a, cv::Mat b);
	void From4DoubleTo3Float(const cv::Mat &in, cv::Mat &out);
}

#endif