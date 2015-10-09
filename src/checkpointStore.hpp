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
        const std::string rawImagesPath;
        const std::string stitcherDumpPath;
        const std::string ringMapPath;
        const std::string exposureMapPath;
        const std::string defaultExtension = ".bmp";
        int c;
    public:
        CheckpointStore(std::string basePath) :
            basePath(basePath),
            rawImagesPath(basePath + "raw_images/"),
            stitcherDumpPath(basePath + "dump/"),
            ringMapPath(basePath + "rings.json"),
            exposureMapPath(basePath + "exposure.json"),
            c(0) { }
        
        void SaveRectifiedImage(InputImageP image);
        
        void SaveStitcherTemporaryImage(Image &image);
        
        void SaveStitcherInput(const std::vector<std::vector<InputImageP>> &rings, const std::map<size_t, double> &exposure);

        void LoadStitcherInput(std::vector<std::vector<InputImageP>> &rings, std::map<size_t, double> &exposure);
        
        void Clear();
        
        bool HasUnstitchedRecording();
    };
}

#endif
