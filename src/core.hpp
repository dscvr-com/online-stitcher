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

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


#ifndef OPTONAUT_CORE_HEADER
#define OPTONAUT_CORE_HEADER

namespace optonaut {
	struct Image {
		Mat img;
		Mat extrinsics;
		Mat intrinsics; 
		int id;
		std::string source;

		vector<KeyPoint> features;
		Mat descriptors;
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

#endif