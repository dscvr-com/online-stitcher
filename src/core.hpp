#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include "lib/tinyxml2/tinyxml2.h"
#include "support.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace optonaut {
	struct Image {
		Mat img;
		Mat extrinsics;
		Mat intrinsics; 
		int id;
		std::string source;
	};

	Image* ImageFromFile(string path) {
		assert(StringEndsWith(path, "jpg") || StringEndsWith(path, "JPG"));
		
		Image* result = new Image();
		result->img = imread(path);
		result->source = path;

		path.replace(path.length() - 3, 3, "xml");

		XMLDocument doc;
		doc.LoadFile(path.c_str());

		XMLElement* root = doc.FirstChildElement("imageParameters");

		result->id = ParseInt(root->Attribute("id"));
		result->extrinsics = MatrixFromXml(root->FirstChildElement("extrinsics")->FirstChildElement("matrix"));
		result->intrinsics = MatrixFromXml(root->FirstChildElement("intrinsics")->FirstChildElement("matrix"));
	
		return result;
	}
}