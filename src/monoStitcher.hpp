////////////////////////////////////////////////////////////////////////////////////////
//
// Beem mono stitcher test by Emi
//
// Let there be MSV!
//
//

#include "image.hpp"

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

	typedef std::shared_ptr<StereoImage> StereoImageP;
    class MonoStitcher {
        public:
        StereoImageP CreateStereo(ImageP a, ImageP b, StereoTarget target);
    };
}

#endif
