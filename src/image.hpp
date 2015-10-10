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
        int cols_;
        int rows_;

        public:
		const cv::Mat &data;
	    std::string source;
        const int &cols;
        const int &rows;

        Image() : data_(0, 0, CV_8UC3), cols_(0), rows_(0),
                  data(data_), source(""), cols(cols_), rows(rows_)  {
        }

        Image(cv::Mat in) : data_(in), cols_(in.cols), rows_(in.rows),
                  data(data_), source(""), cols(cols_), rows(rows_)  {
        }

        Image(const Image &ref) : data_(ref.data_), cols_(data_.cols), rows_(data_.rows),
                  data(data_), source(""), cols(cols_), rows(rows_) {
        }

        Image& operator=(Image other) {
            std::swap(this->data_, other.data_);
            std::swap(this->rows_, other.rows_);
            std::swap(this->cols_, other.cols_);
            std::swap(this->source, other.source);
            return *this;
        }

        inline bool IsLoaded() const {
            return data_.cols != 0 && data_.rows != 0;
        }

        inline cv::Size size() const {
            return cv::Size(cols_, rows_);
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
            cols_ = data.cols;
            rows_ = data.rows;

            assert(cols_ != 0 && rows_ != 0);
        }
	};
}


#endif
