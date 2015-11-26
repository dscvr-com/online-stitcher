#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
	
     class Image {

        public:
		cv::Mat data;
        int cols;
        int rows;
	    std::string source;

        Image() : data(0, 0, CV_8UC3),
                  cols(0), rows(0), source("") {
        }

        Image(cv::Mat in) : data(in), 
                  cols(in.cols), rows(in.rows), source("") {
        }

        Image(const Image &ref) : data(ref.data), 
                  cols(data.cols), rows(data.rows), source("") {
        }

        Image& operator=(Image other) {
            std::swap(this->data, other.data);
            std::swap(this->rows, other.rows);
            std::swap(this->cols, other.cols);
            std::swap(this->source, other.source);
            return *this;
        }

        inline bool IsLoaded() const {
            return data.cols != 0 && data.rows != 0;
        }

        inline cv::Size size() const {
            assert(cols != 0 && rows != 0); //Image was never loaded
            return cv::Size(cols, rows);
        }

        inline int type() const {
            return data.type();
        }

        void Unload() {
            data.release();
        }

        void Load(int flags = cv::IMREAD_COLOR) {
            assert(source != "");

            cv::Mat n = cv::imread(source, flags);
            std::swap(data, n);
            cols = data.cols;
            rows = data.rows;

            assert(cols != 0 && rows != 0);
        }
	};

    typedef std::shared_ptr<Image> ImageP;
}


#endif
