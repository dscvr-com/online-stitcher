#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
    
    const int WorkingWidth = 640;
    const int WorkingHeight = 480;

    namespace colorspace {
        const int RGBA = 0;
        const int RGB = 1;
    }

    struct ImageRef {
        void* data;
        int width;
        int height;
        int colorSpace;

        ImageRef() : data(NULL), width(0), height(0), colorSpace(colorspace::RGBA) { }

        void Invalidate() {
            data = NULL;
            width = 0;
            height = 0;
        }
    };

	struct Image {
		cv::Mat img;
        ImageRef dataRef;
		cv::Mat extrinsics;
        cv::Mat offset;
		cv::Mat intrinsics; 
		int id;
		std::string source;

		std::vector<cv::KeyPoint> features;
		cv::Mat descriptors;

        Image()  : img(0, 0, CV_8UC3), extrinsics(4, 4, CV_64F), intrinsics(3, 3, CV_64F), source("Unknown") {
            offset = cv::Mat::eye(4, 4, CV_64F);
        };

        bool IsLoaded() {
            return img.cols != 0 && img.rows != 0;
        }

        void LoadFromDataRef(bool copy = true);
    
        void SaveToDisk();

        void LoadFromDisk(bool removeFile = true);

        void Unload() {
            img.release();
        }
        
        std::string GetFilePath();
	};
    
    typedef std::shared_ptr<Image> ImageP;
}


#endif
