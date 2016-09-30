#include <functional>

#include "../io/inputImage.hpp"
#include "../stitcher/ringStitcher.hpp"
#include "recorderGraph.hpp"
#include "imageSelector.hpp"

#ifndef OPTONAUT_ASYNC_TOLERANT_RING_RECORDER
#define OPTONAUT_ASYNC_TOLERANT_RING_RECORDER

namespace optonaut {
class AsyncTolerantRingRecorder : public ImageSink {
    private:
        AsyncRingStitcher stitcher;
        std::function<void(SelectionInfo)> PushToStitcher; 
        FunctionSink<SelectionInfo> targetFunc;
        ImageSelector selector;
        StitchingResultP result;
        bool isFinished;

    public:
        AsyncTolerantRingRecorder(RecorderGraph &graph, float warperScale = 400) :
            stitcher(
               fun::map<SelectionPoint*, Mat>(graph.GetTargetsById(), 
                   [](const SelectionPoint* x) {
                        return x->extrinsics;
                   }), 
               warperScale, true),
            // Bind selector directly to stitcher 
            PushToStitcher([&](SelectionInfo info) {
                Assert(!isFinished);

                AutoLoad q(info.image);

                auto rectifiedImage = MonoStitcher::RectifySingle(info);

                stitcher.Push(rectifiedImage);
            }),
            targetFunc(PushToStitcher), 
            selector(graph, 
                    targetFunc,
                    Vec3d(M_PI / 8, M_PI / 8, M_PI / 8),
                    false),
            result(nullptr),
            isFinished(false)
        { }

        virtual void Push(InputImageP img) {
            Log << "Received Image.";

            if(isFinished) 
                return;

            Assert(img != nullptr);
            selector.Push(img);
        }

        StitchingResultP GetResult() {
            return result;
        }
           
        // Can be called from any thread, after finish has been called. 
        StitchingResultP Finalize() {
            Assert(result == nullptr);
            Assert(isFinished);
            
            result = stitcher.Finalize();
            return result;
        }

        // To be called from main thread that also does push.
        virtual void Finish() {
            
            // This is a dirty hack to ignore duplicate calls to finish. 
            if(isFinished)
                return;
            
            Assert(result == nullptr);
            Assert(!isFinished);
            
            selector.Flush();
            isFinished = true;
        }
    };
}

#endif
