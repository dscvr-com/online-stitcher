#include <vector>
#include <opencv2/imgcodecs.hpp>
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
        double ev;
        const ExposureCompensator &exposure;

        cv::Ptr<cv::detail::Blender> blender;
        cv::Ptr<cv::detail::RotationWarper> warper;
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
		bool compensate = false;
		float workScale = 0.2f;
		float warperScale = 1200;

        AsyncRingStitcher(const InputImageP firstImage, 
                std::vector<cv::Mat> rotations, 
                const ExposureCompensator &exposure, 
                double ev = 0);

        void Push(const InputImageP image);

        StitchingResultP Finalize();
};

//Fast pure R-Matrix based stitcher
class RingStitcher {
	public:
    StitchingResultP Stitch(const std::vector<InputImageP> &images, const ExposureCompensator &exposure, ProgressCallback &progress, double ev = 0) {

        std::vector<Mat> rotations = fun::map<InputImageP, Mat>(images, 
                [](const InputImageP &i) { return i->adjustedExtrinsics; }); 

        AsyncRingStitcher core(images[0], rotations, exposure, ev);

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
