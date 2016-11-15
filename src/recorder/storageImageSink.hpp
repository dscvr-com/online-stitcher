#include "../stereo/monoStitcher.hpp"
#include "../io/checkpointStore.hpp"
#include "../common/sink.hpp"

#ifndef OPTONAUT_STORAGE_IMAGE_SINK_HEADER
#define OPTONAUT_STORAGE_IMAGE_SINK_HEADER

namespace optonaut {

    /*
     * Implementation of StereoSink that saves recorder output to 
     * a checkpoint store (and thus usually to disk). 
     */
    class StorageImageSink {

    private:
        CheckpointStore imageStore;
        
    public:
        void Push(SelectionInfo in) {
            imageStore.SaveRectifiedImage(in.image);
        }


        void Finish(std::vector<std::vector<InputImageP>> &postRings, 
                            const std::map<size_t, double> &gains) {
             Log << "Finished";
             imageStore.SaveStitcherInput(postRings, gains);
             Log << "after imageStore Save Stitcher Input";
        }


        StorageImageSink(CheckpointStore &imageStore ) : 
            imageStore(imageStore){

        }
	};
}

#endif
