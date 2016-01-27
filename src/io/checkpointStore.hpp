#include <string>
#include <vector>
#include <map>

#include "../stitcher/stitchingResult.hpp"

#include "inputImage.hpp"

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
        
        static CheckpointStore* DebugStore;
        
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
        
        virtual void SaveRectifiedImage(InputImageP image);
        
        virtual void SaveStitcherTemporaryImage(Image &image);
        
        virtual void SaveStitcherInput(const std::vector<std::vector<InputImageP>> &rings, const std::map<size_t, double> &exposure);
        
        virtual void SaveRing(int ringId, StitchingResultP image);
        virtual void SaveRingMask(int ringId, StitchingResultP image);
        virtual StitchingResultP LoadRing(int ringId);
        
        virtual void SaveOptograph(StitchingResultP image);
        virtual StitchingResultP LoadOptograph();

        virtual void LoadStitcherInput(std::vector<std::vector<InputImageP>> &rings, std::map<size_t, double> &exposure);

        virtual void SaveRingAdjustment(const std::vector<int> &vals);
        virtual void LoadRingAdjustment(std::vector<int> &vals);
        
        virtual void Clear();
        
        virtual bool HasUnstitchedRecording();

        virtual bool SupportsPaging() { return true; }
    };

    class DummyCheckpointStore : public CheckpointStore {
        public:
        DummyCheckpointStore() : CheckpointStore("", "") { }
        virtual void SaveRectifiedImage(InputImageP) { }
        virtual void SaveStitcherTemporaryImage(Image &) { }
        virtual void SaveStitcherInput(const std::vector<std::vector<InputImageP>> &, const std::map<size_t, double> &) { }
        virtual void SaveRing(int, StitchingResultP) { }
        virtual void SaveRingMask(int, StitchingResultP) { }
        virtual StitchingResultP LoadRing(int) { return NULL; }
        virtual void SaveOptograph(StitchingResultP) { }
        virtual StitchingResultP LoadOptograph() { return NULL; }
        virtual void LoadStitcherInput(std::vector<std::vector<InputImageP>> &, std::map<size_t, double> &) { }
        virtual void SaveRingAdjustment(const std::vector<int> &) { }
        virtual void LoadRingAdjustment(std::vector<int> &) { }
        virtual void Clear() { }
        virtual bool HasUnstitchedRecording() { return false; }
        virtual bool SupportsPaging() { return false; }
    };
}

#endif
