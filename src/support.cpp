#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include "support.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace tinyxml2;

namespace optonaut {

	bool StringEndsWith(const string& a, const string& b) {
	    if (b.size() > a.size()) return false;
	    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
	}

	Mat MatrixFromXml(XMLElement* node) {
		int size;
		istringstream(node->Attribute("size")) >> size;

		assert(size == 9 || size == 16);
		int dim = size == 9 ? 3 : 4;

		Mat m(dim, dim, CV_64F);

		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				ostringstream name;
				name << "m" << i << j;
				istringstream text(node->FirstChildElement(name.str().c_str())->GetText());
				text >> m.at<double>(i, j);
			}
		}

		return m;
	}

	int ParseInt(const char* data) {
		int val;
		istringstream text(data);
		text >> val;
		return val;
	}

	string ToString(int i) {
		ostringstream text;
		text << i;
		return text.str();
	}

	void ScaleIntrinsicsToImage(Mat intrinsics, Mat image, Mat &scaled) {
		scaled = Mat::zeros(3, 3, CV_64F);
		
		double scaleFactor = image.cols / (intrinsics.at<double>(0, 2) * 2);
		scaled.at<double>(0, 2) = image.cols / 2;
		scaled.at<double>(1, 2) = image.rows / 2;
		scaled.at<double>(0, 0) = intrinsics.at<double>(0, 0) * scaleFactor;
		scaled.at<double>(1, 1) = intrinsics.at<double>(1, 1) * scaleFactor;
	}

	double GetHorizontalFov(Mat intrinsics) {
		double w = intrinsics.at<double>(0, 2) * 2;
		double f = intrinsics.at<double>(0, 0);

		return 2 * atan2(w / 2, f);
	}

	Mat ExtractRotationVector(Mat r) {
		Mat vec = Mat::zeros(3, 1, CV_64F);

		vec.at<double>(0, 0) = atan2(r.at<double>(2, 1), r.at<double>(2, 2));
		vec.at<double>(1, 0) = atan2(-r.at<double>(2, 0), sqrt(r.at<double>(2, 1) * r.at<double>(2, 1) + r.at<double>(2, 2) * r.at<double>(2, 2)));
		vec.at<double>(2, 0) = atan2(r.at<double>(1, 0), r.at<double>(0, 0));

		return vec;
	}

	double GetAngleOfRotation(Mat r) {
		double t = r.at<double>(0, 0) + r.at<double>(1, 1) + r.at<double>(2, 2);
		return acos((t - 1) / 2);
	}

	double GetDistanceByDimension(Mat a, Mat b, int dim) {
	    float vdata[] = {0, 0, 1, 0};
	    Mat vec(4, 1, CV_64F, vdata);

	    Mat aproj = a * vec;
	    Mat bproj = b * vec;

	    double dist = abs(aproj.at<double>(dim) - bproj.at<double>(dim));
	    dist = asin(dist); 
	    return dist;
	}

	double GetDistanceX(Mat a, Mat b) {
	    return GetDistanceByDimension(a, b, 0);
	}

	double GetDistanceY(Mat a, Mat b) {
	    return GetDistanceByDimension(a, b, 1);
	}

	double GetDistanceZ(Mat a, Mat b) {
	    return GetDistanceByDimension(a, b, 2);
}
}