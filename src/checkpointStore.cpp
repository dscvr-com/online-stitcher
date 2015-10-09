#include "checkpointStore.hpp"
#include "io.hpp"
#include "support.hpp"
#include "dirent.h"
using namespace std;
using namespace cv;

namespace optonaut {
    
    void CheckpointStore::SaveRectifiedImage(InputImageP image) {
        string path = rawImagesPath + ToString(image->id) + ".bmp";
        
        InputImageToFile(image, path);
        image->image.source = path;
    }
    
    void CheckpointStore::SaveStitcherTemporaryImage(Image &image) {
        string path = stitcherDumpPath + ToString(c) + ".bmp";
        
        SaveImage(image, path);
        
        c++;
    }
    
    void CheckpointStore::SaveStitcherInput(const vector<vector<InputImageP>> &rings, const std::map<size_t, double> &exposure) {
        SaveRingMap(rings, ringMapPath);
        SaveExposureMap(exposure, exposureMapPath);
    }
    
    void CheckpointStore::LoadStitcherInput(vector<vector<InputImageP>> &rings, map<size_t, double> &exposure) {
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
        
        exposure = LoadExposureMap(exposureMapPath);
    }
    
    void CheckpointStore::Clear() {
        DeleteDirectories(basePath);
    }
    
    bool CheckpointStore::HasUnstitchedRecording() {
        return IsDirectory(rawImagesPath);
    }
}