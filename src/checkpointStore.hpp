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
        const string stitcherDumpPath;
        const string ringMapPath;
        const string exposureMapPath;
        const string defaultExtension = ".bmp";
        int c;
    public:
        CheckpointStore(string basePath) :
            basePath(basePath),
            rawImagesPath(basePath + "raw_images/"),
            stitcherDumpPath(basePath + "dump/"),
            ringMapPath(basePath + "rings.json"),
            exposureMapPath(basePath + "exposure.json"),
            c(0) { }
        
        void SaveRectifiedImage(InputImageP image) {
            string path = rawImagesPath + ToString(image->id) + ".bmp";
            
            InputImageToFile(image, path);
            image->image.source = path;
        }
        
        void SaveStitcherTemporaryImage(Image &image) {
            string path = stitcherDumpPath + ToString(c) + ".bmp";
           
            SaveImage(image, path);

            c++;
        }
        
        void SaveStitcherInput(const vector<vector<InputImageP>> &rings, const ExposureCompensator &exposure) {
           SaveRingMap(rings, ringMapPath);
           SaveExposureMap(exposure.GetGains(), exposureMapPath);
        }

        void LoadStitcherInput(vector<vector<InputImageP>> &rings, ExposureCompensator &exposure) {
            rings.clear();
            vector<InputImageP> images = LoadAllImagesFromDirectory(rawImagesPath, defaultExtension);
            vector<vector<size_t>> ringmap = LoadRingMap(ringMapPath);
            
            for(auto &r : ringmap) {
                vector<InputImageP> ring;
                for(auto &id : r) {
                    for(auto &image : images) {
                        if((size_t)image->id == id) {
                            ring.push_back(image);
                        }
                    }
                }
                rings.push_back(ring); 
            }

            exposure.SetGains(LoadExposureMap(exposureMapPath));
        }

        void Clear() {
            DeleteDirectories(basePath);
        }

        bool HasUnstitchedRecording() {
            return IsDirectory(rawImagesPath);
        } 
        
    };
}

#endif
