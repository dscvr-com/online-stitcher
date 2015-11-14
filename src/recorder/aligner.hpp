#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "../io/inputImage.hpp"

#ifndef OPTONAUT_ALIGNMENT_HEADER
#define OPTONAUT_ALIGNMENT_HEADER

namespace optonaut {
    struct KeyframeInfo {
        InputImageP keyframe;
        double dist;
    };
    class Aligner {
    public:
		virtual void Push(InputImageP next) = 0;
		virtual cv::Mat GetCurrentBias() const = 0;
        virtual void Dispose() = 0; 
        virtual bool NeedsImageData() = 0;
        virtual void Postprocess(std::vector<InputImageP> imgs) const = 0;
        virtual void Finish() = 0;
        virtual void AddKeyframe(InputImageP next) = 0;
        virtual std::vector<KeyframeInfo> GetClosestKeyframes(const cv::Mat &search, size_t count) const = 0;
        
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
