#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "../common/image.hpp"
#include "../common/progressCallback.hpp"
#include "../common/functional.hpp"
#include "../common/ringProcessor.hpp"
#include "../recorder/exposureCompensator.hpp"
#include "../math/support.hpp"
#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_RSTITCHER_HEADER
#define OPTONAUT_RSTITCHER_HEADER

namespace optonaut {

/*
 * Class capable of efficiently stitching a single ring.  
 */
class AsyncRingStitcher {
    private:
    class Impl;
    // Implementation pointer pattern. 
    Impl* pimpl_;

    float warperScale;
    public:

    /*
     * Creates a new instance of this class.
     *
     * @param firstImage Some image that has equal intrinsics and dimensions to the images that
     *                   will be pushed to this class.
     * @param rotations All expected rotations. The rotations have not to be exactly the same as the rotations that
     *                  are going to be pushed, but they need to cover the same area on the panorama. 
     */
    AsyncRingStitcher(const InputImageP firstImage, 
                std::vector<cv::Mat> rotations, float warperScale = 300, 
                bool fast = true, int roiBuffer = 0);

    /*
     * Pushes an image and adds it to the result. 
     */
    void Push(const InputImageP image);

    /*
     * Gets the warper scale, which relates to the size
     * of the output panorama. 
     * That is the radius of the sphere where the panorama is
     * projected upon. For a warper scale of x, the width of the
     * equirectangular panorama is x * PI * 2 pixels. 
     */
    float GetWarperScale() {
        return warperScale;        
    };

    /*
     * Finalizes and returns the result.
     */
    StitchingResultP Finalize();
    
    /*
     * Frees all allocated resources. 
     */
    ~AsyncRingStitcher();
};

/*
 * Wrapper around AsyncRingStitcher, capable of stitching a ring synchronously
 * and with a progress callback. 
 */
class RingStitcher {
	public:
    StitchingResultP Stitch(const std::vector<InputImageP> &images, ProgressCallback &progress) {

        std::vector<Mat> rotations = fun::map<InputImageP, Mat>(images, 
                [](const InputImageP &i) { return i->adjustedExtrinsics; }); 

        AsyncRingStitcher core(images[0], rotations, GetWarperScale(), false, 0);

        //TODO: Place all IO, exposure compensation and so on here. 

        cout << "images.size() " << images.size() << endl;
        cout << "rotations.size() " << rotations.size() << endl;
        
        for(size_t i = 0; i < images.size(); i++) {
            progress(i / (float)images.size());
            cout << "[stitch] ImageSource" << images[i]->image.source << endl;
            cout << "[stitch] ImageSize" << images[i]->image.size() << endl;
            core.Push(images[i]); 
        }

        return core.Finalize();
    }

    /*
     * Gets the warper scale, which relates to the size
     * of the output panorama. 
     * That is the radius of the sphere where the panorama is
     * projected upon. For a warper scale of x, the width of the
     * equirectangular panorama is x * PI * 2 pixels. 
     */
    float GetWarperScale() {
        // use a fixed scale of 1200. That's approximately
        // what we can fit in memory on an iPhone. 
        return 1200;    
    }

    /*
     * Invokes OpenCV's wave correction for a set of matrices. 
     */
    static void PrepareMatrices(const std::vector<InputImageP> &r);
};

}
#endif
