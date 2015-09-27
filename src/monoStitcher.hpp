////////////////////////////////////////////////////////////////////////////////////////
//
// Beem mono stitcher test by Emi
//
// Let there be MSV!
//
//

#include "image.hpp"
#include "recorderController.hpp"
#include "exposureCompensator.hpp"

#ifndef OPTONAUT_MONO_STITCH_HEADER
#define OPTONAUT_MONO_STITCH_HEADER

namespace optonaut {
    struct StereoTarget {
        cv::Mat center;
        cv::Mat corners[4];
    };

	struct StereoImage {
		ImageP A;
		ImageP B;
		cv::Mat extrinsics; //Center
		bool valid;

		StereoImage() : A(new Image()), B(new Image()), extrinsics(4, 4, CV_64F), valid(false) { }
	};

    class MonoStitcher {
        private: 
        public:
            MonoStitcher() { }
            void CreateStereo(const SelectionInfo &a, const SelectionInfo &b, const SelectionEdge &target, StereoImage &stereo);
    };
}

#endif
