#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../common/image.hpp"
#include "../common/sink.hpp"

#ifndef OPTONAUT_INPUT_IMAGE_HEADER
#define OPTONAUT_INPUT_IMAGE_HEADER

namespace optonaut {
    const int WorkingHeight = 1280;
    const int WorkingWidth = 720;

    /*
     * Represents different colorspaces. 
     */ 
    namespace colorspace {
        const int RGBA = 0;
        const int RGB = 1;
        const int BGRA = 2;
    }

    /*
     * Reference to an image that's in a memory region not controlled
     * by the stitcher code. 
     */
    struct InputImageRef {
        void* data; // Pointer to the data
        int width; // Width of the image in pixels
        int height; // Height of the image in pixels
        int colorSpace; // Pixel format of the image. 

        InputImageRef() : data(NULL), width(0), height(0), 
        colorSpace(colorspace::RGBA) { }

        void Invalidate() {
            data = NULL;
            width = 0;
            height = 0;
        }
    };
   
    /*
     * Represents camera gains. 
     */ 
    struct Gains {
        double red;
        double green;
        double blue;
        
        Gains() : red(1), green(1), blue(1) { }
    };
   
    /*
     * Represents the exposure that was used to capture an image. 
     */ 
    struct ExposureInfo {
        int iso;
        double exposureTime;
        Gains gains;
        
        ExposureInfo() : iso(0), exposureTime(0) { }
    };
   
    /*
     * Represents an input image, that is being fed into the stitching pipeline. 
     */  
	struct InputImage {
        // The image container. 
		Image image;
        // A reference to the external image data, for deferred loading. 
        InputImageRef dataRef;
        // The original rotation data, from the sensor. 
		cv::Mat originalExtrinsics;
        // A copy of the original data, or otherwise, the adjusted sensor data.
        cv::Mat adjustedExtrinsics;
        // The intrinsics data associated with this image. 
		cv::Mat intrinsics;
        // Information about the image exposure. 
        ExposureInfo exposureInfo;
        // The image ID. Unique. 
		int id;

        InputImage() : originalExtrinsics(4, 4, CV_64F), adjustedExtrinsics(4, 4, CV_64F), intrinsics(3, 3, CV_64F) {
        }

        /*
         * True, if the image is loaded into the memory space of
         * the stitching code. 
         */
        bool IsLoaded() {
            return image.data.cols != 0 && image.data.rows != 0;
        }

        /*
         * Loads the image from the external data reference to 
         * the stitching code memory. Takes care of image format and dimensions, if
         * necessary.  
         *
         * Only use this if you exactly know what you are doing. 
         *
         * @param copy If set to false, creates a wrapper around the given memory 
         * ref to save performance, if possible. 
         */
        void LoadFromDataRef(bool copy = true);
	};
   
    /*
     * Shared pointer alias for the InputImage class. 
     */ 
    typedef std::shared_ptr<InputImage> InputImageP;
   
    /* 
     * Creates a downsampled copy. 
     */ 
    InputImageP CloneAndDownsample(InputImageP image);

    /*
     * Autoload construct. Loads an image, if it was not loaded on construction. 
     * On destruction Frees memory, if this autoload did load the image. 
     *
     * Usage example:
     * {
     *      AutoLoad q(image);
     *      -- Image is loaded here --
     * }
     * -- Image is unloaded here, if it was loaded by q --
     */
    class AutoLoad {
        private:
            const InputImageP image;
            bool didLoad;
        public: 
            AutoLoad(const InputImageP &image) : image(image), didLoad(false) {
                if(!image->image.IsLoaded()) {
                    didLoad = true;
                    image->image.Load();
                }
            }

            ~AutoLoad() {
                if(didLoad) {
                    image->image.Unload();
                }
            }
    };
    
    typedef Sink<InputImageP> ImageSink;

}


#endif
