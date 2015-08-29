#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "image.hpp"
#include "support.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "support.hpp"
#include "pipeline.hpp"

using namespace std;
using namespace cv;

namespace optonaut {

    //This method knows too much over the system. Change.

    void Image::LoadFromDataRef(bool copy) {
        assert(!IsLoaded());
        assert(dataRef.data != NULL);
        
        bool isExpectingPortrait = IsPortrait(intrinsics);
        
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
        
        if(img.cols != WorkingWidth && img.rows != WorkingHeight) {
            cv::resize(img, img, cv::Size(WorkingWidth, WorkingHeight));
        }
        
        if(isExpectingPortrait) {
            cv::flip(img, img, 0);
            img = img.t();
        }

        //InvalidateDataRef afterwards.
        dataRef.data = NULL;
        
    }

    string Image::GetFilePath() {
        return Pipeline::tempDirectory + "/" + ToString(id) + ".bmp";
    }
    
    void Image::SaveToDisk() {
        assert(IsLoaded());
        cout << "Saving image " << id << ", width: " << img.cols << ", height: " << img.rows << endl;
        imwrite(GetFilePath(), img); 
        Unload();
    }

    void Image::LoadFromDisk(bool removeFile) {
        assert(!IsLoaded());
        img = imread(GetFilePath());
        cout << "Loading image " << id << ", width: " << img.cols << ", height: " << img.rows << endl;
        if(removeFile) { 
            std::remove(GetFilePath().c_str());
        }
        assert(IsLoaded());
    }
}
