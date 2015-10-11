#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {
	
     class Image {

        private:
		cv::Mat data_;

        public:
		const cv::Mat &data;
        int cols;
        int rows;
	    std::string source;

        Image() : data_(0, 0, CV_8UC3),
                  data(data_), cols(0), rows(0), source("") {
        }

        Image(cv::Mat in) : data_(in), 
                  data(data_), cols(in.cols), rows(in.rows), source("") {
        }

        Image(const Image &ref) : data_(ref.data_), 
                  data(data_), cols(data_.cols), rows(data_.rows), source("") {
        }

        Image& operator=(Image other) {
            std::swap(this->data_, other.data_);
            std::swap(this->rows, other.rows);
            std::swap(this->cols, other.cols);
            std::swap(this->source, other.source);
            return *this;
        }

        inline bool IsLoaded() const {
            return data_.cols != 0 && data_.rows != 0;
        }

        inline cv::Size size() const {
            assert(cols != 0 && rows != 0); //Image was never loaded
            return cv::Size(cols, rows);
        }

        inline int type() const {
            return data_.type();
        }

        void Unload() {
            data_.release();
        }

        void Load(int flags = cv::IMREAD_COLOR) {
            assert(source != "");

            cv::Mat n = cv::imread(source, flags);
            std::swap(data_, n);
            cols = data.cols;
            rows = data.rows;

            assert(cols != 0 && rows != 0);
        }
	};
}


#endif
