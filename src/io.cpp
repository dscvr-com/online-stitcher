#include <algorithm>
#include <string>
#include "lib/tinyxml2/tinyxml2.h"
#include "lib/rapidjson/document.h"
#include "lib/rapidjson/filereadstream.h"
#include "support.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "core.hpp"
#include <fstream>

using namespace cv;
using namespace std;
using namespace tinyxml2;
using namespace rapidjson;

namespace optonaut {

	bool StringEndsWith(const string& a, const string& b) {
	    if (b.size() > a.size()) return false;
	    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
	}

	int MatrixFromXml(XMLElement* node, Mat &out) {
		int size;
		istringstream(node->Attribute("size")) >> size;

		assert(size == 9 || size == 16);
		int dim = size == 9 ? 3 : 4;

		out = Mat(dim, dim, CV_64F);

		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				ostringstream name;
				name << "m" << i << j;
				istringstream text(node->FirstChildElement(name.str().c_str())->GetText());
				text >> out.at<double>(i, j);
			}
		}

		return dim;
	}

	int MatrixFromJson(Value& matrix, Mat &out) {
		assert(matrix.IsArray());

		int size = matrix.Size();

		assert(size == 9 || size == 16);
		int dim = size == 9 ? 3 : 4;

		out = Mat(dim, dim, CV_64F);

		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				out.at<double>(i, j) = matrix[i * dim + j].GetDouble();
			}
		}		

		return dim;
	}

 	void ParseXml(string path, Image* result) {
		XMLDocument doc;
		doc.LoadFile(path.c_str());

		XMLElement* root = doc.FirstChildElement("imageParameters");

		result->id = ParseInt(root->Attribute("id"));
		assert(MatrixFromXml(root->FirstChildElement("intrinsics")->FirstChildElement("matrix"), result->intrinsics) == 3);
		assert(MatrixFromXml(root->FirstChildElement("extrinsics")->FirstChildElement("matrix"), result->extrinsics) == 4);
		result->extrinsics = result->extrinsics.inv();
 	}

 	void ParseJson(string path, Image* result) {
		FILE* fileRef = fopen(path.c_str(), "rb");
		assert(fileRef != NULL);
		char buffer[65536];
		FileReadStream file(fileRef, buffer, sizeof(buffer));
		Document doc;
		doc.ParseStream<0>(file);
		
		result->id = doc["id"].GetInt();
		assert(MatrixFromJson(doc["intrinsics"], result->intrinsics) == 3);
		assert(MatrixFromJson(doc["extrinsics"], result->extrinsics) == 4);
		result->extrinsics = result->extrinsics.inv();
		
		fclose(fileRef);
 	}

 	bool FileExists(const string &fileName)
	{
	    std::ifstream infile(fileName);
	    return infile.good();
	}

	Image* ImageFromFile(string path) {
		assert(StringEndsWith(path, "jpg") || StringEndsWith(path, "JPG"));
		
		Image* result = new Image();
		result->img = imread(path);

		result->source = path;

		string pathWithoutExtensions = path.substr(0, path.length() - 4);
		string xmlPath = pathWithoutExtensions + ".xml";
		string jsonPath = pathWithoutExtensions + ".json";

		if(FileExists(xmlPath)) {
			ParseXml(xmlPath, result);
		} else if(FileExists(jsonPath)) {
			ParseJson(jsonPath, result);
		} else {
			cout << "Unable to open parameter file for " << path << endl;
			assert(false);
		}

		return result;
	}

	void BufferFromBinFile(unsigned char buf[], int *len, string file) {

	    ifstream input(file, std::ios::binary);

	    int i;
	    for(i = 0; i < *len && input.good(); i++) {
	        buf[i] = input.get();
	    }
	    *len = i;
	}
}