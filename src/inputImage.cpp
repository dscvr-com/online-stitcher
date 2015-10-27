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
        
        bool expectingPortrait = IsPortrait(intrinsics);
        
        if(dataRef.colorSpace == colorspace::RGBA) {
            Mat result(dataRef.height, dataRef.width, CV_8UC3);
            
            cv::cvtColor(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC4, dataRef.data), 
                    result,
                    cv::COLOR_RGBA2RGB);
            
            image = Image(result);
            
            copy = false;

        } else if (dataRef.colorSpace == colorspace::RGB) {
            
            image = Image(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC3, dataRef.data));
            

        } else {
            assert(false); //Unsupported input color space.
        }
        
        //We were expeciting portrait but got landscape
        if(expectingPortrait && image.rows < image.cols) {
            assert(false); //Don't use portrait mode
            Mat res;
            cv::flip(image.data, res, 0);
            image = Image(res.t());
            
            copy = false;
        }
        
        if(image.data.cols != WorkingWidth && image.data.rows != WorkingHeight) {
            Mat res;
            cv::resize(image.data, res, cv::Size(WorkingWidth, WorkingHeight));
            image = Image(res);
            
            copy = false;
        }
        
        if(copy) {
            image = Image(image.data.clone());
        }

        //InvalidateDataRef afterwards.
        dataRef.data = NULL;
        
    }
}
