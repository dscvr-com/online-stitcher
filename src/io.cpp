#include <algorithm>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "lib/rapidjson/document.h"
#include "lib/rapidjson/filereadstream.h"
#include "lib/rapidjson/filewritestream.h"
#include "lib/rapidjson/reader.h"
#include "lib/rapidjson/writer.h"
#include "support.hpp"
#include "image.hpp"
#include "inputImage.hpp"
#include "dirent.h"

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
    
    void ReadJsonDocument(Document &doc, const string &path) {
        char buffer[65536];

        FILE* fileRef = fopen(path.c_str(), "r");
        if(fileRef == NULL) {
            cout << "Unable to open data file " << path << endl;
            assert(false);
        }
        
        FileReadStream file(fileRef, buffer, sizeof(buffer));
        doc.ParseStream<0>(file);
        fclose(fileRef);
    }
    
    void WriteJsonDocument(Document &doc, const string &path) {
        FILE* fileRef = fopen(path.c_str(), "w+"); 
        char buffer[65536];
        FileWriteStream os(fileRef, buffer, sizeof(buffer));
        Writer<FileWriteStream> writer(os);
        doc.Accept(writer);
        fclose(fileRef);
    }

 	void ParseInputImageInfoFile(const string &path, InputImageP result) {
        Document doc;
        
        ReadJsonDocument(doc, path);
        
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
 	}
    
    void WriteInputImageInfoFile(const string &path, InputImageP result) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        ReadJsonDocument(doc, path);
        
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
    }

 	bool FileExists(const string &fileName) {
	    std::ifstream infile(fileName);
	    return infile.good();
	}
    
    string GetDataFilePath(const string &imagePath) {
        assert(StringEndsWith(imagePath, ".jpg") || StringEndsWith(imagePath, ".bmp"));
        
        string pathWithoutExtensions = imagePath.substr(0, imagePath.length() - 4);
        string jsonPath = pathWithoutExtensions + ".json";
        
        return jsonPath;
    }
    
    void InputImageToFile(InputImageP image, const string &path) {
        string jsonPath = GetDataFilePath(path);
        
        WriteInputImageInfoFile(jsonPath, image);
        
        imwrite(path, image->image.data);
    }
    

    InputImageP InputImageFromFile(const string &path, bool shallow) {
        string jsonPath = GetDataFilePath(path);
		
		InputImageP result(new InputImage());
        
		result->image.source = path;
        if(!shallow) {
            result->image.Load();
        }

        ParseInputImageInfoFile(jsonPath, result);

		return result;
	}

	void BufferFromBinFile(unsigned char buf[], size_t len, const string &file) {

	    ifstream input(file, std::ios::binary);

	    size_t i;
	    for(i = 0; i < len && input.good(); i++) {
	        buf[i] = input.get();
	    }
	    assert(i == len);
	}

	void BufferToBinFile(unsigned char buf[], size_t len, const string &file) {
	    FILE* d;
		d = fopen(file.c_str(), "w");
		fwrite(buf, 1, len, d);
		fclose(d);
	}

    void SaveExposureMap(const std::map<size_t, double> &exposure, const string &path) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        Value jExposure(kArrayType);

        for(auto &e : exposure) {
            Value je(kObjectType);

            je.AddMember("id", (uint64_t)e.first, allocator);
            je.AddMember("e", e.second, allocator);

            jExposure.PushBack(je, allocator);
        }

        doc.AddMember("exposure", jExposure, allocator);

        WriteJsonDocument(doc, path);
    }

    std::map<size_t, double> LoadExposureMap(const string &path) {
        Document doc;

        std::map<size_t, double> exposure;
    
        ReadJsonDocument(doc, path);

        Value &jExposure = doc["exposure"];

        for(SizeType i = 0; i < jExposure.Size(); i++) {
            exposure.at(jExposure[i]["id"].GetUint64()) = 
                jExposure[i]["e"].GetDouble();
        }

        return exposure;
    }

    void SaveRingMap(const vector<vector<InputImageP>> &rings, const string &path) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        Value jRings(kArrayType);

        for(auto &ring : rings) {
            Value jRing(kArrayType);

            for(auto &img : ring) {
                jRing.PushBack(img->id, allocator);
            }

            jRings.PushBack(jRing, allocator);
        }

        doc.AddMember("rings", jRings, allocator);

        WriteJsonDocument(doc, path);
    }

    vector<vector<size_t>> LoadRingMap(const string &path) {
        Document doc;

        vector<vector<size_t>> rings;
    
        ReadJsonDocument(doc, path);

        Value &jRings = doc["rings"];

        for(SizeType i = 0; i < jRings.Size(); i++) {
            vector<size_t> ring;
            for(SizeType j = 0; j < jRings[i].Size(); j++) {
                ring.push_back(jRings[i][j].GetUint64());
            }
            rings.push_back(ring);
        }

        return rings;
    }
        
    vector<InputImageP> LoadAllImagesFromDirectory(const string &path, const string &extension) {
        vector<InputImageP> images;
        
        DIR *dir;
        struct dirent *ent;
        if((dir = opendir(path.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                string name = ent->d_name;
                
                if(StringEndsWith(name, extension)) {
                    images.push_back(InputImageFromFile(name, true));
                }
                
            }
            closedir(dir);
        } else {
            //Could not open dir.
            assert(false);
        }
        
        return images;
    }
}
