#include "../stereo/monoStitcher.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_STORAGE_SINK_HEADER
#define OPTONAUT_STORAGE_SINK_HEADER

namespace optonaut {

    /*
     * Implementation of StereoSink that saves recorder output to 
     * a checkpoint store (and thus usually to disk). 
     */
    class ImageSink {

    private:
        CheckpointStore imageStore;
        
    public:
        virtual void Push(SelectionInfo in) {
            imageStore.SaveRectifiedImage(in.image);
        }

        virtual void Finish(std::vector<std::vector<InputImageP>> &postRings, 
                            const std::map<size_t, double> &gains) {
            
             imageStore.SaveStitcherInput(postRings, gains);



        }


        ImageSink(CheckpointStore &imageStore ) : 
            imageStore(imageStore){

        }
	};
}

#endif
