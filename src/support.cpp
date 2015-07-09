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

	void MatrixFromXml(XMLElement* node, Mat &out) {
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

		out = m.clone();
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

	void ScaleIntrinsicsToImage(Mat intrinsics, Mat image, Mat &scaled, double fupscaling) {
		scaled = Mat::zeros(3, 3, CV_64F);
		
		double scaleFactor = image.cols / (intrinsics.at<double>(0, 2) * 2);
		scaled.at<double>(0, 2) = image.cols / 2;
		scaled.at<double>(1, 2) = image.rows / 2;
		//Todo: Remove factor 10 - only for debug. 
		scaled.at<double>(0, 0) = intrinsics.at<double>(0, 0) * scaleFactor * fupscaling;
		scaled.at<double>(1, 1) = intrinsics.at<double>(1, 1) * scaleFactor * fupscaling;
		scaled.at<double>(2, 2) = 1;
	}

	double GetHorizontalFov(Mat intrinsics) {
		double w = intrinsics.at<double>(0, 2) * 2;
		double f = intrinsics.at<double>(0, 0);

		return 2 * atan2(w / 2, f);
	}

	void ExtractRotationVector(Mat r, Mat &v) {
		Mat vec = Mat::zeros(3, 1, CV_64F);

		vec.at<double>(0, 0) = atan2(r.at<double>(2, 1), r.at<double>(2, 2));
		vec.at<double>(1, 0) = atan2(-r.at<double>(2, 0), sqrt(r.at<double>(2, 1) * r.at<double>(2, 1) + r.at<double>(2, 2) * r.at<double>(2, 2)));
		vec.at<double>(2, 0) = atan2(r.at<double>(1, 0), r.at<double>(0, 0));

		v = vec.clone();
	}
	void CreateRotationZ(double a, Mat &t) {
		double v[] = {
			cos(a), -sin(a), 0, 0,
			sin(a), cos(a),  0, 0,
			0, 	    0,       1, 0,
			0,      0,       0, 1 
		};
		Mat rot(4, 4, CV_64F, v);

		t = rot.clone();
	}

	void CreateRotationX(double a, Mat &t) {
		double v[] = {
			1, 0,      0,       0,
			0, cos(a), -sin(a), 0,
			0, sin(a), cos(a),  0,
			0, 0,      0,       1 
		};
		Mat rot(4, 4, CV_64F, v);

		t = rot.clone();
	}

	void CreateRotationY(double a, Mat &t) {
		double v[] = {
			cos(a),  0, sin(a), 0,
			0, 	     1, 0,      0,
			-sin(a), 0, cos(a), 0,
			0,       0, 0,      1 
		};
		Mat rot(4, 4, CV_64F, v);

		cout << "rot" << rot << endl;

		t = rot.clone();
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

	void From4DoubleTo3Float(const Mat &in, Mat &out) {
		out = Mat::zeros(3, 3, CV_32F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<float>(i, j) = (float)in.at<double>(i, j);
			}
		}
	}
	void From3FloatTo4Double(const Mat &in, Mat &out) {
		out = Mat::zeros(4, 4, CV_64F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<double>(i, j) = (double)in.at<float>(i, j);
			}
		}
		out.at<double>(3, 3) = 1;
	}
}