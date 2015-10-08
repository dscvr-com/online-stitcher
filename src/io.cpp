#include <algorithm>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "lib/rapidjson/document.h"
#include "lib/rapidjson/filereadstream.h"
#include "support.hpp"
#include "image.hpp"

using namespace cv;
using namespace std;
using namespace rapidjson;

namespace optonaut {

	bool StringEndsWith(const string& a, const string& b) {
	    if (b.size() > a.size()) return false;
	    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
	}

    int IdFromFileName(const string &in) {
        size_t l = in.find_last_of("/");
        if(l == string::npos) {
            return ParseInt(in.substr(0, in.length() - 4));
        } else {
            return ParseInt(in.substr(l + 1, in.length() - 5 - l));
        }
    }

	int MatrixFromJson(const Value& matrix, Mat &out) {
		assert(matrix.IsArray());

		int size = matrix.Size();

        int dim = sqrt(size);
        
        //Check for quadratical matrix.
        assert(size == dim * dim);

		out = Mat(dim, dim, CV_64F);

		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				out.at<double>(i, j) = matrix[i * dim + j].GetDouble();
			}
		}		

		return dim;
	}
    
    int MatrixToJson(const Mat &in, Value& out, Document::AllocatorType& allocator) {
        assert(in.cols == in.rows); //Only square matrices supported.
        
        int dim = in.cols;
        
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                out.PushBack(in.at<double>(i, j), allocator);
            }
        }
        
        return dim;
    }
    
    char buffer[65536];
    FILE* fileRef = NULL;
    
    void OpenJsonDocument(Document &doc, const string &path, const string &flags) {
        assert(fileRef == NULL);
        FILE* fileRef = fopen(path.c_str(), flags.c_str());
        if(fileRef == NULL) {
            cout << "Unable to open data file " << path << endl;
            assert(false);
        }
        
        FileReadStream file(fileRef, buffer, sizeof(buffer));
        doc.ParseStream<0>(file);
    }
    
    void CloseJsonDocument() {
        assert(fileRef != NULL);
        fclose(fileRef);
        fileRef = NULL;
    }

 	void ParseInputImageInfoFile(const string &path, InputImageP result) {
        Document doc;
        
        OpenJsonDocument(doc, path, "rb");
        
		result->id = doc["id"].GetInt();
		assert(MatrixFromJson(doc["intrinsics"], result->intrinsics) == 3);
        if(doc.HasMember("originalExtrinsics")) {
            assert(MatrixFromJson(doc["originalExtrinsics"],
                                  result->originalExtrinsics) == 4);
            assert(MatrixFromJson(doc["adjustedExtrinsics"],
                                  result->adjustedExtrinsics) == 4);
        } else {
            assert(MatrixFromJson(doc["extrinsics"], result->adjustedExtrinsics) == 4);
        }
		
        CloseJsonDocument();
 	}
    
    void WriteInputImageInfoFile(const string &path, InputImageP result) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        OpenJsonDocument(doc, path, "wx");
        
        doc.AddMember("id", result->id, allocator);
        MatrixToJson(result->intrinsics,
                     doc.AddMember("intrinsics", kArrayType, allocator),
                     allocator);
        MatrixToJson(result->adjustedExtrinsics,
                     doc.AddMember("adjustedExtrinsics", kArrayType, allocator),
                     allocator);
        MatrixToJson(result->originalExtrinsics,
                     doc.AddMember("originalExtrinsics", kArrayType, allocator),
                     allocator);
        
        CloseJsonDocument();
    }

 	bool FileExists(const string &fileName)
	{
	    std::ifstream infile(fileName);
	    return infile.good();
	}
    
    string GetDataFilePath(const string &imagePath) {
        assert(StringEndsWith(imagePath, ".jpg") || StringEndsWith(imagePath, ".bmp"));
        
        string pathWithoutExtensions = imagePath.substr(0, imagePath.length() - 4);
        string jsonPath = pathWithoutExtensions + ".json";
        
        return jsonPath;
    }
    
    void InputImageToFile(ImageP image, const string &path) {
        string jsonPath = GetDataFilePath(path);
        
        WriteImageInfoFile(jsonPath, image);
        
        imwrite(path, image->image->data);
    }

    InputImageP InputImageFromFile(const string &path, bool shallow) {
        string jsonPath = GetDataFilePath(path);
		
		ImageP result(new Image());
        
		result->image->source = path;
        if(!shallow) {
            LoadImageData(result->image);
        }

        ParseImageInfoFile(jsonPath, result);

		return result;
	}

    void LoadImageData(Image &image) {
        image->data = imread(image.source);
    }
    
	void BufferFromBinFile(unsigned char buf[], size_t len, string file) {

	    ifstream input(file, std::ios::binary);

	    size_t i;
	    for(i = 0; i < len && input.good(); i++) {
	        buf[i] = input.get();
	    }
	    assert(i == len);
	}

	void BufferToBinFile(unsigned char buf[], size_t len, string file) {
	    FILE* d;
		d = fopen(file.c_str(), "w");
		fwrite(buf, 1, len, d);
		fclose(d);
	}
}
