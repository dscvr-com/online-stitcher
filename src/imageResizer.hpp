#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

#include "image.hpp"
#include "support.hpp"
#include "imageSelector.hpp"

#ifndef OPTONAUT_RESIZER_HEADER
#define OPTONAUT_RESIZER_HEADER

namespace optonaut {
class ImageResizer {

    int configuration;

public:
    ImageResizer(int selectorConfiguration) : configuration(selectorConfiguration) { }

    void Resize(Mat &image) {
        int w = 4096;
        int h = 4096;
        
        if(configuration == ImageSelector::ModeCenter) {
            int ih = 1280;

            Mat canvas(w, h, CV_8UC3);
            canvas.setTo(Scalar::all(0));

            Mat resized(w, ih, CV_8UC3);
            resize(image, resized, cv::Size(w, ih));
            int rowStart = (h - ih) / 2;
            
            resized.copyTo(canvas.rowRange(rowStart, rowStart + ih));

            image = canvas;
        } else if (configuration == ImageSelector::ModeAll) {
            Mat resized(w, h, CV_8UC3);
            resize(image, resized, cv::Size(w, h));
            image = resized;
        } else {
            cout << "Invalid selector mode" << endl;
            assert(false);
        }
         
    }
};
}
#endif
