#include "../stereo/monoStitcher.hpp"
#include "stereoSink.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_STORAGE_SINK_HEADER
#define OPTONAUT_STORAGE_SINK_HEADER

namespace optonaut {

    /*
     * Implementation of StereoSink that saves recorder output to 
     * a checkpoint store (and thus usually to disk). 
     */
    class StorageSink : public StereoSink {

    private:
        CheckpointStore leftStore;
        CheckpointStore rightStore;
        
    public:
        virtual void Push(StereoImage stereo) {
            leftStore.SaveRectifiedImage(stereo.A);
            rightStore.SaveRectifiedImage(stereo.B);
            
            stereo.A->image.Unload();
            stereo.B->image.Unload();
        }

        virtual void Finish(std::vector<std::vector<InputImageP>> &leftRings, 
                            std::vector<std::vector<InputImageP>> &rightRings,
                            const std::map<size_t, double> &gains) {
            
            leftStore.SaveStitcherInput(leftRings, gains);
            rightStore.SaveStitcherInput(rightRings, gains); 
        }

        StorageSink(CheckpointStore &leftStore, CheckpointStore &rightStore) : 
            leftStore(leftStore), rightStore(rightStore) {

        }
	};
}

#endif
