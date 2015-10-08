#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "image.hpp"
#include "io.hpp"
#include "support.hpp"
#include "dirent.h"
#include "image.hpp"
#include "stitchingResult.hpp"

#ifndef OPTONAUT_CHECKPOINT_HEADER
#define OPTONAUT_CHECKPOINT_HEADER

using namespace std;
using namespace cv;

namespace optonaut {
    class CheckpointStore {
    private:
        const string basePath;
        const string rawImagesPath;
        const string defaultExtension = ".bmp";
        int c;
    public:
        CheckpointStore(string basePath) :
            basePath(basePath),
            rawImagesPath(basePath + "raw_images/"),
            c(0) { }
        
        void SaveRawImage(ImageP image) {
            string path = rawImagesPath + ToString(c) + ".bmp";
            
            ImageToFile(image, path);
            
            c++;
        }
        
        //Loads an a mat for an unloaded ImageP
        void LoadRawImageData(ImageP image) {
            
        }
        
        void SaveRing(StitchingResultP ring, int idx) {
            
        }
        
        void LoadRing(StitchingResultP ring, int idx) {
            
        }
        
        vector<ImageP> LoadShallowRawImages() {
            vector<ImageP> images;
            
            DIR *dir;
            struct dirent *ent;
            if ((dir = opendir (rawImagesPath.c_str())) != NULL) {
                while ((ent = readdir (dir)) != NULL) {
                    string name = ent->d_name;
                    
                    if(StringEndsWith(name, defaultExtension)) {
                        images.push_back(ImageFromFile(name));
                    }
                    
                }
                closedir (dir);
            } else {
                //Could not open dir.
                assert(false);
            }
            
            return images;
        }

       void LoadFromDisk(size_t id, cv::Mat &img, int loadFlags) {
            img = imread(GetFilePath(id), loadFlags);
            assert(img.cols != 0 && img.rows != 0);
       }
            
       void SaveToDisk(size_t id, cv::Mat &img) {
            imwrite(GetFilePath(id), img); 
       }
        
        void SaveToDisk() {
            assert(IsLoaded());
            //cout << "Saving image " << id << ", width: " << img.cols << ", height: " << img.rows << endl;
            Image::SaveToDisk((size_t)this, img);
            Unload();
        }
    };
}

#endif
