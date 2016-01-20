#include "../common/support.hpp"

#include "checkpointStore.hpp"
#include "io.hpp"
#include "dirent.h"

using namespace std;
using namespace cv;

namespace optonaut {
    
    void CheckpointStore::SaveRectifiedImage(InputImageP image) {
        string path = rawImagesPath + ToString(image->id) + defaultExtension;
        
        InputImageToFile(image, path);
        image->image.source = path;
    }
    
    void CheckpointStore::SaveStitcherTemporaryImage(Image &image) {
        string path = stitcherDumpPath + ToString(c) + defaultExtension;
        
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
        
    void CheckpointStore::SaveRingAdjustment(const std::vector<int> &vals) {
        SaveIntList(vals, ringAdjustmentPath); 
    }

    void CheckpointStore::LoadRingAdjustment(std::vector<int> &vals) {
        if(FileExists(ringAdjustmentPath)) {
            vals = LoadIntList(ringAdjustmentPath);
        }
    }

    void CheckpointStore::SaveRing(int ringId, StitchingResultP image) {
        StitchingResultToFile(image, ringPath + "ring_" + ToString(ringId), defaultExtension);
    }
    
    void CheckpointStore::SaveRingMask(int ringId, StitchingResultP image) {
        StitchingResultToFile(image, ringPath + "ring_" + ToString(ringId), defaultExtension, true);
    }
    
    StitchingResultP CheckpointStore::LoadRing(int ringId) {
        return StitchingResultFromFile(ringPath + "ring_" + ToString(ringId), defaultExtension);
    }
    
    void CheckpointStore::SaveOptograph(StitchingResultP image) {
        StitchingResultToFile(image, optographPath + "result", defaultExtension);
    }
    
    StitchingResultP CheckpointStore::LoadOptograph() {
        return StitchingResultFromFile(optographPath + "result", defaultExtension);
    }
    
    void CheckpointStore::Clear() {
        DeleteDirectories(basePath);
    }
    
    bool CheckpointStore::HasUnstitchedRecording() {
        return IsDirectory(rawImagesPath);
    }
}
