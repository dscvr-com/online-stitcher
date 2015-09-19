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
        
        void Push(ImageP image) {
            current = image->originalExtrinsics.clone();
        }
        
        void Dispose() {
            // *muha*
        }
        
        Mat GetCurrentRotation() const {
            return current;
        }
        
        void Postprocess(vector<ImageP>) const { };
    };
}
#endif
