#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "../common/image.hpp"

#ifndef OPTONAUT_INPUT_IMAGE_HEADER
#define OPTONAUT_INPUT_IMAGE_HEADER

namespace optonaut {
    const int WorkingHeight = 1280;
    const int WorkingWidth = 720;

    namespace colorspace {
        const int RGBA = 0;
        const int RGB = 1;
    }

    struct InputImageRef {
        void* data;
        int width;
        int height;
        int colorSpace;

        InputImageRef() : data(NULL), width(0), height(0), colorSpace(colorspace::RGBA) { }

        void Invalidate() {
            data = NULL;
            width = 0;
            height = 0;
        }
    };
    
    struct Gains {
        double red;
        double green;
        double blue;
        
        Gains() : red(1), green(1), blue(1) { }
    };
    
    struct ExposureInfo {
        int iso;
        double exposureTime;
        Gains gains;
        
        ExposureInfo() : iso(0), exposureTime(0) { }
    };
    
	struct InputImage {
		Image image;
        InputImageRef dataRef;
		cv::Mat originalExtrinsics;
        cv::Mat adjustedExtrinsics;
		cv::Mat intrinsics;
        ExposureInfo exposureInfo;
		int id;
        double vtag;

        InputImage() : originalExtrinsics(4, 4, CV_64F), adjustedExtrinsics(4, 4, CV_64F), intrinsics(3, 3, CV_64F), vtag(0) {
        }

        bool IsLoaded() {
            return image.data.cols != 0 && image.data.rows != 0;
        }

        void LoadFromDataRef(bool copy = true);
	};
    
    typedef std::shared_ptr<InputImage> InputImageP;
    
    InputImageP CloneAndDownsample(InputImageP image);
}


#endif
