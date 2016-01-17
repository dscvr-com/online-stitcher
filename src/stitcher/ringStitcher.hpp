#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../common/image.hpp"
#include "../common/progressCallback.hpp"
#include "../common/functional.hpp"
#include "../common/ringProcessor.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "../math/support.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_RSTITCHER_HEADER
#define OPTONAUT_RSTITCHER_HEADER

namespace optonaut {

class AsyncRingStitcher {
    private:
    class Impl;
    Impl* pimpl_;
    public:

    AsyncRingStitcher(const InputImageP firstImage, 
                std::vector<cv::Mat> rotations, float warperScale = 300, 
                bool fast = true, int roiBuffer = 0);

    void Push(const InputImageP image);

    StitchingResultP Finalize();
    
    ~AsyncRingStitcher();
};

//Fast pure R-Matrix based stitcher
class RingStitcher {
	public:
    StitchingResultP Stitch(const std::vector<InputImageP> &images, ProgressCallback &progress) {

        std::vector<Mat> rotations = fun::map<InputImageP, Mat>(images, 
                [](const InputImageP &i) { return i->adjustedExtrinsics; }); 

        AsyncRingStitcher core(images[0], rotations, 1200, false, 0);

        //TODO: Place all IO, exposure compensation and so on here. 
        
        for(size_t i = 0; i < images.size(); i++) {
            progress(i / (float)images.size());
            core.Push(images[i]); 
        }

        return core.Finalize();
    }

    static void PrepareMatrices(const std::vector<InputImageP> &r);
};

}
#endif
