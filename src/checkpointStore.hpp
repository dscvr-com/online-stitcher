#include <string>
#include <vector>
#include <map>
#include "inputImage.hpp"
#include "stitchingResult.hpp"

#ifndef OPTONAUT_CHECKPOINT_HEADER
#define OPTONAUT_CHECKPOINT_HEADER

namespace optonaut {
    class CheckpointStore {
    private:
        const std::string basePath;
        const std::string sharedPath;
        const std::string rawImagesPath;
        const std::string stitcherDumpPath;
        const std::string ringMapPath;
        const std::string ringPath;
        const std::string optographPath;
        const std::string exposureMapPath;
        const std::string defaultExtension = ".bmp";
        const std::string ringAdjustmentPath;
        int c;
    public:
        CheckpointStore(std::string basePath, std::string sharedPath) :
            basePath(basePath),
            sharedPath(sharedPath),
            rawImagesPath(basePath + "raw_images/"),
            stitcherDumpPath(basePath + "dump/"),
            ringMapPath(basePath + "rings.json"),
            ringPath(basePath + "rings/"),
            optographPath(basePath + "optograph/"),
            exposureMapPath(basePath + "exposure.json"),
            ringAdjustmentPath(sharedPath + "offsets.json"),
            c(0) { }
        
        void SaveRectifiedImage(InputImageP image);
        
        void SaveStitcherTemporaryImage(Image &image);
        
        void SaveStitcherInput(const std::vector<std::vector<InputImageP>> &rings, const std::map<size_t, double> &exposure);
        
        void SaveRing(int ringId, StitchingResultP image);
        void SaveRingMask(int ringId, StitchingResultP image);
        StitchingResultP LoadRing(int ringId);
        
        void SaveOptograph(StitchingResultP image);
        StitchingResultP LoadOptograph();

        void LoadStitcherInput(std::vector<std::vector<InputImageP>> &rings, std::map<size_t, double> &exposure);

        void SaveRingAdjustment(const std::vector<int> &vals);
        void LoadRingAdjustment(std::vector<int> &vals);
        
        void Clear();
        
        bool HasUnstitchedRecording();
    };
}

#endif
