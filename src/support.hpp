#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

namespace optonaut {

	bool StringEndsWith(const std::string& a, const std::string& b);
	void MatrixFromXml(tinyxml2::XMLElement* node, cv::Mat &out);
	int ParseInt(const char* data);

	std::string ToString(int i);

	void ScaleIntrinsicsToImage(cv::Mat intrinsics, cv::Mat image, cv::Mat &scaled, double fupscaling = 1);

	double GetHorizontalFov(cv::Mat intrinsics);

	void ExtractRotationVector(cv::Mat r, cv::Mat &vec);
	double GetAngleOfRotation(cv::Mat r);
	void CreateRotationZ(double radians, cv::Mat &out);
	void CreateRotationX(double radians, cv::Mat &out);
	void CreateRotationY(double radians, cv::Mat &out);

	double GetDistanceByDimension(cv::Mat a, cv::Mat b, int dim);
	double GetDistanceX(cv::Mat a, cv::Mat b);
	double GetDistanceY(cv::Mat a, cv::Mat b);
	double GetDistanceZ(cv::Mat a, cv::Mat b);
	void From4DoubleTo3Float(const cv::Mat &in, cv::Mat &out);
	void From3FloatTo4Double(const cv::Mat &in, cv::Mat &out);
}

#endif