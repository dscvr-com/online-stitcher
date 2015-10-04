#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "aligner.hpp"
#include "pairwiseVisualAligner.hpp"
#include "stat.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_BUNDLE_ALIGNMENT_HEADER
#define OPTONAUT_BUNDLE_ALIGNMENT_HEADER

namespace optonaut {
	class BundleAligner : public Aligner {
	private:
		PairwiseVisualAligner visual;
        vector<ImageP> images;
    public: 
		BundleAligner() : visual() { }

        bool NeedsImageData() {
            return true;
        }

        void Dispose() {

        }

		void Push(ImageP next) {
            for(auto img : images) {
                if(GetAngleOfRotation(img->originalExtrinsics, next->originalExtrinsics) < M_PI / 2) {
                //if(visual.AreOverlapping(next, img)) {
                     visual.FindCorrespondence(next, img);
                     visual.FindCorrespondence(img, next);
                }
            } 
            images.push_back(next);
        }

		Mat GetCurrentRotation() const {
			return images[images.size() - 1]->originalExtrinsics;
		}

        void Postprocess(vector<ImageP> imgs) const {
            visual.RunBundleAdjustment(imgs);
        };
	};
}

#endif
