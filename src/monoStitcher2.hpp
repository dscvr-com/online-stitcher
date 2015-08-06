////////////////////////////////////////////////////////////////////////////////////////
//
// Beem mono stitcher test by Emi
//
// Let there be MSV!
//
//

#include "core.hpp"

#ifndef OPTONAUT_MONO_STITCH2_HEADER
#define OPTONAUT_MONO_STITCH2_HEADER

namespace optonaut {

	struct StereoImage {
		Image A;
		Image B;
		cv::Mat extrinsics; //Center
		bool valid;

		StereoImage() : valid(false) { }
	};

	StereoImage *CreateStereo(Image *a, Image *b);
}

#endif