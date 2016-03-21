#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "../io/inputImage.hpp"

#ifndef OPTONAUT_ALIGNMENT_HEADER
#define OPTONAUT_ALIGNMENT_HEADER

namespace optonaut {
    /*
     * Encapsulates a keyframe (e.g. an image with known position) and the distance to that keyframe.
     */
    struct KeyframeInfo {
        InputImageP keyframe;
        double dist;
    };

    /*
     * Aligner interface. Designed as a class that follows an asynchronous/streaming pattern. 
     */
    class Aligner {
    public:

        /*
         * Sends an input image to this aligner. 
         */
		virtual void Push(InputImageP next) = 0;

        /*
         * Returns the current sensor bias, that is the difference between 
         * the value received from the device sensors and the actual value. 
         */
		virtual cv::Mat GetCurrentBias() const = 0;

        /*
         * Frees all resources allocated by the aligner. 
         */
        virtual void Dispose() = 0; 

        /*
         * True if image data has to be available for the next image, false if not. 
         */
        virtual bool NeedsImageData() = 0;

        /*
         * Runs postprocessing steps on all given images. This has to be called when recording is finished. 
         */
        virtual void Postprocess(std::vector<InputImageP> imgs) const = 0;

        /*
         * Finishes all asynchronous operations without releasing memory resources. 
         */
        virtual void Finish() = 0;

        /*
         * Manually adds a keyframe (an image with a known position) to the aligner. 
         */
        virtual void AddKeyframe(InputImageP next) = 0;

        /*
         * Gets the keyframes which are closest to the rotation given by search. 
         */
        virtual std::vector<KeyframeInfo> GetClosestKeyframes(const cv::Mat &search, size_t count) const = 0;
        virtual double GetCurrentVtag() const {
            return 0;
        }
        
        /* 
         * Gets the keyframe closest do the rotation given by search. 
         */
        InputImageP GetClosestKeyframe(const cv::Mat &search) const {
            auto keyframes = GetClosestKeyframes(search, 1);
            if(keyframes.size() != 1) {
                return NULL;
            } else {
                return keyframes[0].keyframe;
            }
        }
	};
}

#endif
