#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

namespace optonaut {

	bool MatIs(const cv::Mat &in, int rows, int cols, int type);
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
	void From3DoubleTo3Float(const cv::Mat &in, cv::Mat &out);
	void From3FloatTo4Double(const cv::Mat &in, cv::Mat &out);
	void From3DoubleTo4Double(const cv::Mat &in, cv::Mat &out);
	bool ContainsNaN(const cv::Mat &in);

	double min2(double a, double b);
	double max2(double a, double b);
	double min4(double a, double b, double c, double d);
	double max4(double a, double b, double c, double d);
	double angleAvg(double x, double y);
	double interpolate(double x, double x1, double x2, double y1, double y2);
}

#endif