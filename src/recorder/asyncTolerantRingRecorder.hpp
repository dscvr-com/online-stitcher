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
        AsyncTolerantRingRecorder(const SelectionInfo &firstImage, 
                RecorderGraph &graph, float warperScale = 400) :
            // For initialization, take extrinsics from graph
            stitcher(MonoStitcher::RectifySingle(firstImage),
                   fun::map<SelectionPoint*, Mat>(graph.GetTargetsById(), 
                       [](const SelectionPoint* x) {
                            return x->extrinsics;
                       }), 
                   warperScale, true),
            // Bind selector directly to stitcher 
            selector(graph, 
                    std::bind(&AsyncTolerantRingRecorder::PushToStitcher, 
                        this, 
                        std::placeholders::_1),
                    Vec3d(M_PI / 8, M_PI / 8, M_PI / 8),
                    false)
        {

        }

        void PushToStitcher(SelectionInfo info) {
            AutoLoad q(info.image);
            stitcher.Push(MonoStitcher::RectifySingle(info));
            //stitcher.Push(info.image);
        }

        void Push(const InputImageP img) {
            Assert(img != nullptr);
            selector.Push(img);
        }

        StitchingResultP Finalize() {
            selector.Flush();
            return stitcher.Finalize();
        }
};
}

#endif
