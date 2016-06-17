#include <algorithm>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include <sys/stat.h> 
#include <sys/types.h>
#include <unistd.h>
#include <errno.h> 

#include "../lib/rapidjson/document.h"
#include "../lib/rapidjson/filereadstream.h"
#include "../lib/rapidjson/filewritestream.h"
#include "../lib/rapidjson/reader.h"
#include "../lib/rapidjson/writer.h"

#include "../common/support.hpp"
#include "../common/image.hpp"
#include "../common/logger.hpp"
#include "../stitcher/stitchingResult.hpp"

#include "inputImage.hpp"
#include "dirent.h"

using namespace cv;
using namespace std;
using namespace rapidjson;

//NOTE: All directories given as params have to end with a slash. 

namespace optonaut {

    bool IsDirectory(const std::string &path) {
        struct stat info;

        if(stat(path.c_str(), &info) != 0)
            return false;
        else if(info.st_mode & S_IFDIR) 
            return true;
    
        return false;
    }

    // Please kill me. 
    void _mkdir(const char *dir) {
        char tmp[512];
        char *p = NULL;
        size_t len;

        snprintf(tmp, sizeof(tmp),"%s",dir);
        len = strlen(tmp);
        if(tmp[len - 1] == '/')
            tmp[len - 1] = 0;
        for(p = tmp + 1; *p; p++)
        if(*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
        mkdir(tmp, S_IRWXU);
    }

    // Plase kill me too.
    int remove_directory(const char *path) {
        DIR *d = opendir(path);
        size_t path_len = strlen(path);
        int r = -1;

        if (d) {
            struct dirent *p;

            r = 0;

            while (!r && (p=readdir(d))) {
                int r2 = -1;
                char *buf;
                size_t len;

                /* Skip the names "." and ".." as we don't want to recurse on them. */
                if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
                    continue;
                }

                len = path_len + strlen(p->d_name) + 2; 
                buf = (char*)malloc(len);

                if (buf) {
                    struct stat statbuf;

                    snprintf(buf, len, "%s/%s", path, p->d_name);

                    if (!stat(buf, &statbuf)) {
                        if (S_ISDIR(statbuf.st_mode)) {
                            r2 = remove_directory(buf);
                        }
                        else {
                            r2 = unlink(buf);
                        }
                    }
                    free(buf);
                }
                r = r2;
            }
            closedir(d);
        }

        if (!r) {
            r = rmdir(path);
        }

        return r;
    }

    void CreateDirectories(const string &path) {
        string dir = GetDirectoryName(path);

        if(dir == "")
            return;

       assert(path.length() < 512);
       _mkdir(dir.c_str());
    }

    void DeleteDirectories(const string &path) {
        remove_directory(path.c_str());
    } 

    int IdFromFileName(const string &in) {
        size_t l = in.find_last_of("/");
        if(l == string::npos) {
            return ParseInt(in.substr(0, in.length() - 4));
        } else {
            return ParseInt(in.substr(l + 1, in.length() - 5 - l));
        }
    }
    
    void SaveImage(Image &image, const std::string &path) {
        CreateDirectories(path);
        imwrite(path, image.data);
        image.source = path;
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
        out.SetArray();
        
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
            cout << "Unable to read data file " << path << endl;
            assert(false);
        }
        
        FileReadStream file(fileRef, buffer, sizeof(buffer));
        doc.ParseStream<0>(file);
        fclose(fileRef);
    }
    
    void WriteJsonDocument(Document &doc, const string &path) {

        CreateDirectories(path);

        FILE* fileRef = fopen(path.c_str(), "w+");
        if(fileRef == NULL) {
            cout << "Unable to write data file " << path << endl;
            assert(false);
        }
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
            //Internal temporary image format
            assert(MatrixFromJson(doc["originalExtrinsics"],
                                  result->originalExtrinsics) == 4);
            assert(MatrixFromJson(doc["adjustedExtrinsics"],
                                  result->adjustedExtrinsics) == 4);
            result->image.cols = doc["width"].GetInt();
            result->image.rows = doc["height"].GetInt();
        } else {
            //External debug format
            assert(MatrixFromJson(doc["extrinsics"], result->originalExtrinsics) == 4);
            result->adjustedExtrinsics = result->originalExtrinsics.clone();
        }
 	}
    
    void WriteInputImageInfoFile(const string &path, InputImageP result) {
        
        CreateDirectories(path);

        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        doc.SetObject();
        doc.AddMember("id", result->id, allocator);
        doc.AddMember("width", result->image.cols, allocator);
        doc.AddMember("height", result->image.rows, allocator);

        Value intrinsics; 
        MatrixToJson(result->intrinsics,
                     intrinsics,
                     allocator);
        doc.AddMember("intrinsics", intrinsics, allocator);

        Value adjustedExtrinsics; 
        MatrixToJson(result->adjustedExtrinsics,
                     adjustedExtrinsics,
                     allocator);
        doc.AddMember("adjustedExtrinsics", adjustedExtrinsics, allocator);

        Value originalExtrinsics; 
        MatrixToJson(result->originalExtrinsics,
                     originalExtrinsics,
                     allocator);
        doc.AddMember("originalExtrinsics", originalExtrinsics, allocator);

        WriteJsonDocument(doc, path);
    }
    
    void WriteStitchingResultInfoFile(const string &path, StitchingResultP result) {
        
        CreateDirectories(path);
        
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        doc.SetObject();
        doc.AddMember("x", result->corner.x, allocator);
        doc.AddMember("y", result->corner.y, allocator);
        doc.AddMember("id", result->id, allocator);
        doc.AddMember("seamed", result->seamed, allocator);
        doc.AddMember("width", result->image.cols, allocator);
        doc.AddMember("height", result->image.rows, allocator);
        
        WriteJsonDocument(doc, path);
    }
    
    void ParseStitchingResultInfoFile(const string &path, StitchingResultP result, int &width, int &height) {
        Document doc;
        
        ReadJsonDocument(doc, path);
        
        result->corner = cv::Point(doc["x"].GetInt(), doc["y"].GetInt());
        result->id = doc["id"].GetInt();
        result->seamed = doc["seamed"].GetBool();
        width = doc["width"].GetInt();
        height = doc["height"].GetInt();
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
    
    void StitchingResultToFile(StitchingResultP image, const string &path, const string &extension, bool maskOnly) {
        string infoFilePath = path + ".data.json";
        string imagePath = path + ".image" + extension;
        string maskPath = path + ".mask" + extension;
        
        WriteStitchingResultInfoFile(infoFilePath, image);
        
        imwrite(maskPath, image->mask.data);
        image->mask.source = maskPath;
       
        if(!maskOnly) { 
            imwrite(imagePath, image->image.data);
            image->image.source = imagePath;
        }
    }
    
    StitchingResultP StitchingResultFromFile(const string &path, const string &extension) {
        string infoFilePath = path + ".data.json";
        string imagePath = path + ".image" + extension;
        string maskPath = path + ".mask" + extension;
        
        if(!FileExists(infoFilePath))
            return StitchingResultP(NULL);
    
        StitchingResultP res(new StitchingResult());

        int width, height;
        
        ParseStitchingResultInfoFile(infoFilePath, res, width, height);
        res->image = Image(Mat(0, 0, CV_8UC3)); 
        res->mask = Image(Mat(0, 0, CV_8UC3));
        res->image.cols = width;
        res->image.rows = height;
        res->image.source = imagePath;
        res->mask.source = maskPath;
        res->mask.cols = width;
        res->mask.rows = height;

        return res;
    }
    
    
    template <typename T>
    void SaveListGeneric(const std::vector<T> input, const std::string &path) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        doc.SetArray();
        
        for(T &val : input) {
            doc.PushBack(val, allocator);
        }
        
        WriteJsonDocument(doc, path);
    }
    
    template <typename T>
    std::vector<T> LoadListGeneric(const std::string &path) {
        Document doc;
        ReadJsonDocument(doc, path);
        
        assert(doc.IsArray());
        size_t size = doc.Size();
        vector<T> res;
        
        for(size_t i = 0; i < size; i++) {
            //Haha, lol. As if this will go trough the typecheck. 
            res.push_back(doc[i]);
        }
        
        return res;
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
        jExposure.SetArray();

        for(auto &e : exposure) {
            Value je(kObjectType);
            je.SetObject();

            je.AddMember("id", (uint64_t)e.first, allocator);
            je.AddMember("e", e.second, allocator);

            jExposure.PushBack(je, allocator);
        }
        
        doc.SetObject();
        doc.AddMember("exposure", jExposure, allocator);

        WriteJsonDocument(doc, path);
    }

    std::map<size_t, double> LoadExposureMap(const string &path) {
        Document doc;

        std::map<size_t, double> exposure;
    
        ReadJsonDocument(doc, path);

        Value &jExposure = doc["exposure"];

        for(SizeType i = 0; i < jExposure.Size(); i++) {
            exposure[(size_t)(jExposure[i]["id"].GetUint64())] = 
                jExposure[i]["e"].GetDouble();
        }

        return exposure;
    }

    void SaveRingMap(const vector<vector<InputImageP>> &rings, const string &path) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        Value jRings(kArrayType);
        jRings.SetArray();

        for(auto &ring : rings) {
            Value jRing(kArrayType);
            jRing.SetArray();

            for(auto &img : ring) {
                jRing.PushBack(img->id, allocator);
            }

            jRings.PushBack(jRing, allocator);
        }
        doc.SetObject();
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
                ring.push_back((size_t)jRings[i][j].GetUint64());
            }
            rings.push_back(ring);
        }

        return rings;
    }
    
    void SaveIntList(const std::vector<int> &vals, const std::string &path) {
        Document doc;
        Document::AllocatorType &allocator = doc.GetAllocator();
        
        Value jOffsets(kArrayType);
        jOffsets.SetArray();

        for(auto &val : vals) {
            jOffsets.PushBack(val, allocator);
        }
        doc.SetObject();
        doc.AddMember("offsets", jOffsets, allocator);

        WriteJsonDocument(doc, path);
    }

    std::vector<int> LoadIntList(const std::string &path) {
        Document doc;

        vector<int> vals;
        ReadJsonDocument(doc, path);

        Value &jOffsets = doc["offsets"];

        for(SizeType j = 0; j < jOffsets.Size(); j++) {
            vals.push_back((double)jOffsets[j].GetInt());
        }

        return vals;
    }
        
    vector<InputImageP> LoadAllImagesFromDirectory(const string &path, const string &extension) {
        vector<InputImageP> images;
        
        DIR *dir;
        struct dirent *ent;
        if((dir = opendir(path.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                string name = ent->d_name;
                
                if(StringEndsWith(name, extension)) {
                    images.push_back(InputImageFromFile(path + name, true));
                }
                
            }
            closedir(dir);
        } else {
            //Could not open dir.
            AssertM(false, "Could not open dir: " + path);
        }
        
        return images;
    }
}
