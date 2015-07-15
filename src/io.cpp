#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include "support.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "core.hpp"

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

	Image* ImageFromFile(string path) {
		assert(StringEndsWith(path, "jpg") || StringEndsWith(path, "JPG"));
		
		Image* result = new Image();
		result->img = imread(path);

		//TODO: That's only correct for certain cases!
		//flip(result->img, result->img, -1);

		result->source = path;

		path.replace(path.length() - 3, 3, "xml");

		XMLDocument doc;
		doc.LoadFile(path.c_str());

		XMLElement* root = doc.FirstChildElement("imageParameters");

		result->id = ParseInt(root->Attribute("id"));
		MatrixFromXml(root->FirstChildElement("extrinsics")->FirstChildElement("matrix"), result->extrinsics);
		result->extrinsics = result->extrinsics.inv();
		MatrixFromXml(root->FirstChildElement("intrinsics")->FirstChildElement("matrix"), result->intrinsics);
	
		return result;
	}

}