#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "inputImage.hpp"
#include "support.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace optonaut {

    void InputImage::LoadFromDataRef(bool copy) {
        assert(!IsLoaded());
        assert(dataRef.data != NULL);
        
        bool isExpectingPortrait = IsPortrait(intrinsics);
        
        if(dataRef.colorSpace == colorspace::RGBA) {
            
            cv::cvtColor(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC4, dataRef.data), 
                    image.data, 
                    cv::COLOR_RGBA2RGB);

        } else if (dataRef.colorSpace == colorspace::RGB) {
            
            image = Image(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC3, dataRef.data));
            if(copy) {
                image = Image(image.data.clone());
            }

        } else {
            assert(false);
        }
        
        if(image.data.cols != WorkingWidth && image.data.rows != WorkingHeight) {
            Mat res;
            cv::resize(image.data, res, cv::Size(WorkingWidth, WorkingHeight));
            image = Image(res);
        }
        
        if(isExpectingPortrait) {
            Mat res;
            cv::flip(image.data, res, 0);
            image = Image(res.t());
        }

        //InvalidateDataRef afterwards.
        dataRef.data = NULL;
        
    }
}
