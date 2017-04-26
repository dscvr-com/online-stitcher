#include "../io/inputImage.hpp"
#include "../io/io.hpp"
#include "../io/checkpointStore.hpp"
#include "../stereo/monoStitcher.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/functional.hpp"
#include "../common/queueProcessor.hpp"
#include "../common/asyncQueueWorker.hpp"
#include "../common/static_timer.hpp"
#include "../common/progressCallback.hpp"
#include "../common/reduceProcessor.hpp"
#include "../stitcher/simpleSphereStitcher.hpp"
#include "../stitcher/ringStitcher.hpp"
#include "../debug/debugHook.hpp"
#include "../common/logger.hpp"
#include "recorderGraph.hpp"
#include "recorderGraphGenerator.hpp"
#include "trivialAligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include "asyncTolerantRingRecorder.hpp"
#include "ringCloser.hpp"
#include "visualStabilizer.hpp"

#ifndef OPTONAUT_RECORDER_HEADER
#define OPTONAUT_RECORDER_HEADER

// Legacy class, containing some intrinsics and headers. 

namespace optonaut {
    
    struct StereoPair {
        SelectionInfo a;
        SelectionInfo b;
    };
    
    class Recorder {
    public:
        static Mat androidBase;
        static Mat iosBase;
        static Mat iosZero;
        static Mat androidZero;

        static string tempDirectory;
        static string version;
    };    
}

#endif
