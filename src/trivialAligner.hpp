#include <opencv2/opencv.hpp>

#include "support.hpp"
#include "aligner.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_TRIVIAL_ALIGNMENT_HEADER
#define OPTONAUT_TRIVIAL_ALIGNMENT_HEADER

namespace optonaut {
    class TrivialAligner : public Aligner {
    private:
        Mat current;
    public:
        TrivialAligner() : current(Mat::eye(4, 4, CV_64F)) { }
        
        bool NeedsImageData() {
            return false;
        }
        
        void Push(InputImageP image) {
            current = image->originalExtrinsics.clone();
            image->adjustedExtrinsics = image->originalExtrinsics;
        }
        
        void Dispose() {
            // *muha*
        }
        
        Mat GetCurrentRotation() const {
            return current;
        }
        
        void AddKeyframe(InputImageP next) { }
        std::vector<KeyframeInfo> GetClosestKeyframes(const cv::Mat &search, size_t count) const { return { }; }
        
        void Postprocess(vector<InputImageP>) const { };
        void Finish() { };
    };
}
#endif
