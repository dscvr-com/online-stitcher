#include <string>
#include <cmath>
#include <vector>
#include <map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "inputImage.hpp"

using namespace std;
using namespace cv;

#include "../math/support.hpp"
#include "../common/assert.hpp"
#include "../common/static_timer.hpp"

namespace optonaut {

    void InputImage::LoadFromDataRef(bool copy) {
        
        Log << dataRef.width << "x" << dataRef.height;

        // This method should be tuned. A lot. 
        // Some parts are protected by asserts on purpose, since they should
        // not be used in production. 
        
        // STimer loadTimer(true);
        
        Assert(!IsLoaded());
        Assert(dataRef.data != NULL);
        
        bool expectingPortrait = IsPortrait(intrinsics);
        //loadTimer.Tick("## IsPortrait");
        
        if(dataRef.colorSpace == colorspace::RGBA) {
            STimer loadTimer(true);
            Mat result(dataRef.height, dataRef.width, CV_8UC3);
            //loadTimer.Tick("## Allocate Mat");
            cv::cvtColor(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC4, dataRef.data), 
                    result,
                         cv::COLOR_RGBA2RGB);
            //loadTimer.Tick("## cvtColor");
            
            image = Image(result);
            
            copy = false;
            //loadTimer.Tick("## mkImage");

        } else if(dataRef.colorSpace == colorspace::BGRA) {
            STimer loadTimer(true);
            Mat result(dataRef.height, dataRef.width, CV_8UC3);
            //loadTimer.Tick("## Allocate Mat");
            cv::cvtColor(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC4, dataRef.data),
                    result,
                    cv::COLOR_BGRA2RGB);
            //loadTimer.Tick("## cvtColor");

            image = Image(result);

            copy = false;
            //loadTimer.Tick("## mkImage");

        } else if (dataRef.colorSpace == colorspace::RGB) {
            image = Image(
                    cv::Mat(dataRef.height, dataRef.width, CV_8UC3, dataRef.data));
        } else {
            Assert(false); //Unsupported input color space.
        }
        
        //We were expeciting portrait but got landscape
        if(expectingPortrait && image.rows < image.cols) {
            AssertFalseInProduction(false); //Don't use portrait mode
            Mat res;
            cv::flip(image.data, res, 0);
            image = Image(res.t());
            
            copy = false;
        }
        
        if(image.data.cols != WorkingWidth && image.data.rows != WorkingHeight) {
            AssertM(abs((float)image.cols / image.rows - (float)WorkingWidth / (float)WorkingHeight) < 0.01, "Aspect ratio does match for resize.");
            Mat res;
            cv::resize(image.data, res, cv::Size(WorkingWidth, WorkingHeight));
            image = Image(res);
            
            copy = false;
        }
        
        if(copy) {
            AssertFalseInProduction(false);
            image = Image(image.data.clone());
        }

        //InvalidateDataRef afterwards.
        dataRef.data = NULL;
    }
    
    
    InputImageP CloneAndDownsample(InputImageP image) {
        InputImageP clone(new InputImage());
        clone->originalExtrinsics = image->originalExtrinsics.clone();
        clone->adjustedExtrinsics = image->adjustedExtrinsics.clone();
        clone->intrinsics = image->intrinsics.clone();
        clone->exposureInfo = image->exposureInfo;
        clone->id = image->id;
        
        Mat downscaled;
        
        AssertM(image->IsLoaded(), "Image is loaded before downsampling");
        
        pyrDown(image->image.data, downscaled);
        
        clone->image = Image(downscaled);
        
        return clone;
    }
}
