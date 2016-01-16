#include "../io/inputImage.hpp"
#include "../stitcher/ringStitcher.hpp"
#include "recorderGraph.hpp"
#include "imageSelector.hpp"

#ifndef OPTONAUT_ASYNC_TOLERANT_RING_RECORDER
#define OPTONAUT_ASYNC_TOLERANT_RING_RECORDER

namespace optonaut {
class AsyncTolerantRingRecorder {
    private:
        AsyncRingStitcher stitcher;
        ImageSelector selector;
    public:
        AsyncTolerantRingRecorder(const InputImageP &firstImage, 
                RecorderGraph &graph, float warperScale = 400) :
            // For initialization, take extrinsics from grap
            stitcher(firstImage,
                   fun::map<SelectionPoint*, Mat>(graph.GetTargetsById(), 
                       [](auto x) {
                            return x->extrinsics;
                       }), 
                   warperScale),
            // Bind selector directly to stitcher 
            selector(graph, 
                    std::bind(&AsyncTolerantRingRecorder::PushToStitcher, 
                        this, 
                        std::placeholders::_1),
                    M_PI / 16,
                    true)
        {

        }

        void PushToStitcher(const SelectionInfo info) {
            stitcher.Push(info.image);
        }

        void Push(const InputImageP img) {
            selector.Push(img);
        }

        StitchingResultP Finalize() {
            return stitcher.Finalize();
        }
};
}

#endif
