#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace tinyxml2;

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

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

		Mat m(dim, dim, CV_32F);

		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				ostringstream name;
				name << "m" << i << j;
				istringstream text(node->FirstChildElement(name.str().c_str())->GetText());
				text >> m.at<float>(i, j);
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
}

#endif