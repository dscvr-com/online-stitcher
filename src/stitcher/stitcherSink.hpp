#include "../stereo/monoStitcher.hpp"
#include "../stitcher/multiringStitcher.hpp"

#ifndef OPTONAUT_STITCHER_SINK_HEADER
#define OPTONAUT_STITCHER_SINK_HEADER

namespace optonaut {
    /*
     * Stereo sink that directly stitches results in memory, instead of writing them to disk. 
     */
    class StitcherSink : public StereoSink {

    private:
        StitchingResultP leftResult;
        StitchingResultP rightResult;
        
    public:
        virtual void Push(StereoImage) {

        }

        StitchingResultP GetLeftResult() {
            return leftResult;
        }
       
        StitchingResultP GetRightResult() {
            return leftResult;
        }

        virtual void Finish(std::vector<std::vector<InputImageP>> &leftImages, 
                            std::vector<std::vector<InputImageP>> &rightImages,
                            const std::map<size_t, double> &gains) {

            DummyCheckpointStore store;

            ExposureCompensator exp;
            exp.SetGains(gains);

            MultiRingStitcher leftStitcher(store);
            leftStitcher.InitializeForStitching(leftImages, exp, 0.4);
            leftResult = leftStitcher.Stitch(ProgressCallback::Empty);

            MultiRingStitcher rightStitcher(store);
            rightStitcher.InitializeForStitching(rightImages, exp, 0.4);
            rightResult = rightStitcher.Stitch(ProgressCallback::Empty);
        }
	};
}

#endif
