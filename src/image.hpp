#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {

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
		cv::Mat intrinsics; 
		int id;
		std::string source;

		std::vector<cv::KeyPoint> features;
		cv::Mat descriptors;

        Image()  : img(0, 0, CV_8UC3), extrinsics(4, 4, CV_64F), intrinsics(3, 3, CV_64F), source("Unknown") { };

        bool IsLoaded() {
            return img.cols != 0 && img.rows != 0;
        }

        void Load(bool copy = true) {
            assert(!IsLoaded());
            assert(dataRef.data != NULL);

            if(dataRef.colorSpace == colorspace::RGBA) {
                cv::cvtColor(cv::Mat(dataRef.height, dataRef.width, CV_8UC4, dataRef.data), img, cv::COLOR_RGBA2RGB);
            } else if (dataRef.colorSpace == colorspace::RGB) {
                img = cv::Mat(dataRef.height, dataRef.width, CV_8UC3, dataRef.data);
                if(copy) {
                    img = img.clone();
                }
            } else {
                assert(false);
            }

        }

        void Unload() {
            img.release();
        }
	};

	typedef std::shared_ptr<Image> ImageP;
}


#endif
