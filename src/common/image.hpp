#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../common/assert.hpp"

#ifndef OPTONAUT_IMAGE_HEADER
#define OPTONAUT_IMAGE_HEADER

namespace optonaut {

	 /*
      * General purpose wrapper around
      * a cv::Mat that contains an image. 
      */
     class Image {

        public:

        /*
         * Underlying cv::Mat object, holding image data. 
         */
		cv::Mat data;

        /*
         * Dimensions of the image. 
         */
        int cols;
        int rows;

        /*
         * Source of the image in the file system. 
         */
	    std::string source;

        /*
         * Creates a new, empty instance of this class.
         */
        Image() : data(0, 0, CV_8UC3),
                  cols(0), rows(0), source("") {
        }

        /*
         * Creates a new instance from the given mat object. 
         */
        Image(cv::Mat in) : data(in), 
                  cols(in.cols), rows(in.rows), source("") {
        }

        /*
         * Creates a shallow copy from the given Image object. 
         */
        Image(const Image &ref) : data(ref.data), 
                  cols(data.cols), rows(data.rows), source("") {
        }

        /*
         * Assignment copy operator. 
         */
        Image& operator=(Image other) {
            std::swap(this->data, other.data);
            std::swap(this->rows, other.rows);
            std::swap(this->cols, other.cols);
            std::swap(this->source, other.source);
            return *this;
        }

        /*
         * Returns true if the underlying cv::Mat contains data. 
         */
        inline bool IsLoaded() const {
            return data.cols != 0 && data.rows != 0;
        }

        /*
         * Returns the size of this image, regardless if data is loaded. 
         */
        inline cv::Size size() const {
            Assert(cols != 0 && rows != 0); //Image was never loaded
            return cv::Size(cols, rows);
        }

        /*
         * Returns the datatype of the underlying image. 
         */
        inline int type() const {
            return data.type();
        }

        /*
         * Unloads the underlying cv::Mat. Metadata is persisted. 
         */
        void Unload() {
            data.release();
        }

        /*
         * Reloads the image from its source. 
         *
         * @param flags cv::imread loading flags. 
         */
        void Load(int flags = cv::IMREAD_COLOR) {
            AssertNEQM(source, std::string(""), "Image has source.");

            cv::Mat n = cv::imread(source, flags);
            std::swap(data, n);
            cols = data.cols;
            rows = data.rows;

            Assert(cols != 0 && rows != 0);
        }
	};

    /*
     * Typedef for a shared pointer to Images. 
     */
    typedef std::shared_ptr<Image> ImageP;
}


#endif
