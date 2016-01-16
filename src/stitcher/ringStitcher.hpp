#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../common/image.hpp"
#include "../common/progressCallback.hpp"
#include "../common/functional.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "../math/support.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_RSTITCHER_HEADER
#define OPTONAUT_RSTITCHER_HEADER

namespace optonaut {

class AsyncRingStitcher {
    private:
        size_t n;

        cv::Ptr<cv::detail::Blender> blender;
        cv::Ptr<cv::detail::RotationWarper> warper;
        cv::Ptr<cv::WarperCreator> warperFactory;
        std::vector<cv::Point> corners;
        std::vector<cv::Size> warpedSizes;
        cv::UMat uxmap, uymap;
        cv::Rect resultRoi;
        cv::Rect dstRoi;
        cv::Rect dstCoreMaskRoi;
        RingProcessor<StitchingResultP> queue;
        cv::Mat warpedMask;
        cv::Mat K;
        void FindSeams(const StitchingResultP &a, const StitchingResultP &b);
        void Feed(const StitchingResultP &in);
    public:

        AsyncRingStitcher(const InputImageP firstImage, 
                std::vector<cv::Mat> rotations, float warperScale = 400, 
                int roiBuffer = 100);

        void Push(const InputImageP image);

        StitchingResultP Finalize();
};

//Fast pure R-Matrix based stitcher
class RingStitcher {
	public:
    StitchingResultP Stitch(const std::vector<InputImageP> &images, ProgressCallback &progress) {

        std::vector<Mat> rotations = fun::map<InputImageP, Mat>(images, 
                [](const InputImageP &i) { return i->adjustedExtrinsics; }); 

        AsyncRingStitcher core(images[0], rotations, 1200, 0);

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
