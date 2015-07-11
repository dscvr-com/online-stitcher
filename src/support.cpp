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

	bool MatIs(const Mat &in, int rows, int cols, int type) {
		return in.rows >= rows && in.cols >= cols && in.type() == type;
	}

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
		assert(MatIs(intrinsics, 3, 3, CV_64F));

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
		assert(MatIs(intrinsics, 3, 3, CV_64F));

		double w = intrinsics.at<double>(0, 2) * 2;
		double f = intrinsics.at<double>(0, 0);

		return 2 * atan2(w / 2, f);
	}

	void ExtractRotationVector(Mat r, Mat &v) {
		assert(MatIs(r, 3, 3, CV_64F));

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
		assert(MatIs(r, 3, 3, CV_64F));
		double t = r.at<double>(0, 0) + r.at<double>(1, 1) + r.at<double>(2, 2);
		return acos((t - 1) / 2);
	}

	double GetDistanceByDimension(Mat a, Mat b, int dim) {
		assert(MatIs(a, 4, 4, CV_64F));
		assert(MatIs(b, 4, 4, CV_64F));

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

	//TODO: Cleanup conversion mess. 

	void From4DoubleTo3Float(const Mat &in, Mat &out) {
		assert(MatIs(in, 4, 4, CV_64F));

		out = Mat::zeros(3, 3, CV_32F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<float>(i, j) = (float)in.at<double>(i, j);
			}
		}
	}
	void From3DoubleTo3Float(const Mat &in, Mat &out) {
		assert(MatIs(in, 3, 3, CV_64F));

		out = Mat::zeros(3, 3, CV_32F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<float>(i, j) = (float)in.at<double>(i, j);
			}
		}
	}
	void From3FloatTo4Double(const Mat &in, Mat &out) {
		assert(MatIs(in, 3, 3, CV_32F));

		out = Mat::zeros(4, 4, CV_64F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<double>(i, j) = (double)in.at<float>(i, j);
			}
		}
		out.at<double>(3, 3) = 1;
	}

	bool ContainsNaN(const Mat &in) {
		assert(in.type() == CV_64F);
		for(int i = 0; i < in.rows; i++) {
			for(int j = 0; j < in.cols; j++) {
				if(in.at<double>(i, j) != in.at<double>(i, j)) {
					return true;
				}
			}
		}
		return false;
	}


	void From3DoubleTo4Double(const Mat &in, Mat &out) {
		assert(MatIs(in, 3, 3, CV_64F));

		out = Mat::zeros(4, 4, CV_64F);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				out.at<double>(i, j) = in.at<double>(i, j);
			}
		}
		out.at<double>(3, 3) = 1;
	}

	double min2(double a, double b) {
		return a > b ? b : a;
	}

	double max2(double a, double b) {
		return a < b ? b : a;
	}

	double min4(double a, double b, double c, double d) {
		return min2(min2(a, b), min2(c, d));
	}

	double max4(double a, double b, double c, double d) {
		return max2(max2(a, b), max2(c, d));
	}

	double angleAvg(double x, double y) {
		double r = (((x + 2 * M_PI) + (y + 2 * M_PI)) / 2);

		while(r >= 2 * M_PI) {
			r -= 2 * M_PI;
		}

		return r;
	}

	double interpolate(double x, double x1, double x2, double y1, double y2) {
		return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
	}
}