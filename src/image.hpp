#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
    const int WorkingWidth = 640;
    const int WorkingHeight = 360;

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
		cv::Mat originalExtrinsics;
        cv::Mat adjustedExtrinsics;
		cv::Mat intrinsics; 
		int id;
		std::string source;

		std::vector<cv::KeyPoint> features;
		cv::Mat descriptors;

        Image()  : img(0, 0, CV_8UC3), originalExtrinsics(4, 4, CV_64F), adjustedExtrinsics(4, 4, CV_64F), intrinsics(3, 3, CV_64F), source("Unknown") {
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
         
        static std::string GetFilePath(size_t id);

        static void LoadFromDisk(size_t id, cv::Mat &img, int loadFlags = CV_LOAD_IMAGE_COLOR);
        
        static void SaveToDisk(size_t id, cv::Mat &img);
	};
    
    typedef std::shared_ptr<Image> ImageP;
}


#endif
