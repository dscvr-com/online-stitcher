#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "image.hpp"
#include "io.hpp"
#include "support.hpp"
#include "dirent.h"

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
        
        vector<ImageP> LoadShallowRawImages() {
            vector<ImageP> images;
            
            DIR *dir;
            struct dirent *ent;
            if ((dir = opendir (rawImagesPath)) != NULL) {
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
    };
}

#endif