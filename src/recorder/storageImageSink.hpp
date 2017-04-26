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
    class StorageImageSink : StereoSink {

    private:
        CheckpointStore imageStore;
        std::vector<InputImageP> images;
    public:
        void Push(SelectionInfo in) {
            imageStore.SaveRectifiedImage(in.image);
            in.image->image.Unload();
            images.push(in.image);
        }

        void Finish() {
            Log << "Finished";
        }

        void SaveInputSummary(const RecorderGraph& graph) {
            Log << "Saving input summary";
            const vector<vector<InputImageP>> rings = graph.SplitIntoRings(images);
            const map<size_t, double> dummy; 
            imageStore.SaveStitcherInput(images, dummy) 
        }

        StorageImageSink(CheckpointStore &imageStore ) : 
            imageStore(imageStore){
        }
	};
}

#endif
