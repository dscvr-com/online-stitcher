#include "../common/image.hpp"
#include "../recorder/imageSelector.hpp"
#include "../recorder/exposureCompensator.hpp"

#ifndef OPTONAUT_MONO_STITCH_HEADER
#define OPTONAUT_MONO_STITCH_HEADER

namespace optonaut {

    /*
     * Represents a pair of stereo images. 
     */
	struct StereoImage {
		InputImageP A; // The left image. 
		InputImageP B; // The right image. 
		cv::Mat extrinsics; // The orientation of the image plane on which A and B are projected. 
		bool valid; // True, if this stereo image is valid. 

        // Creates a new instance of this class. 
		StereoImage() : A(new InputImage()), B(new InputImage()), extrinsics(4, 4, CV_64F), valid(false) { }
	};

    /*
     * Class capable of doing stereo rectification, adjusted
     * to the rotational model we use. 
     */
    class MonoStitcher {
        private: 
        public:
            MonoStitcher() { }

            /* 
             * Creates a stereo image from two SelectionInfos, which 
             * pair an image and a selection point.
             *
             * A has to be the point left of B. Both images have to be close 
             * enough together for them to overlap on an the image plane which is 
             * located at the rotational middle between the two selection points. 
             *
             * @param a The first selection point.
             * @param b The second selection point. 
             * @param stereo Stereo image to place the results in. 
             */
            void CreateStereo(const SelectionInfo &a, const SelectionInfo &b, StereoImage &stereo) const;
            
            /*
             * Transforms a single image to match it's given selection point.
             *
             * @param a The image to rectify. 
             * @returns The rectified image. 
             */
            static InputImageP RectifySingle(const SelectionInfo &a);
    };
}

#endif
