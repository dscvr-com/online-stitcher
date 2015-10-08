#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
	
        struct Image {
		cv::Mat data;
		std::string source;

        Image() : img(0, 0, CV_8UC3), source("") {
        }

        bool IsLoaded() {
            return img.cols != 0 && img.rows != 0;
        }

        void Unload() {
            img.release();
        }
	};
    
    typedef std::shared_ptr<Image> ImageP;
}


#endif
