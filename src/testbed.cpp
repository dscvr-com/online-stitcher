#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>

#include "intrinsics.hpp"
#include "pipeline.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;


bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char* argv[]) {



    int n = argc - 1;
    shared_ptr<Pipeline> pipe(NULL);
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    sort(files.begin(), files.end(), CompareByFilename);
  
    for(int i = 0; i < n; i++) {
        auto image = ImageFromFile(files[i]);
        //image->intrinsics = iPhone6Intrinsics;
        
        //Flip image accordingly to portrait. 
       // cv::flip(image->img, image->img, 0);
       // image->img = image->img.t();

        //Print out selector points.
        /*RStitcher stitcher; 
        stitcher.blendMode = cv::detail::Blender::FEATHER;
        
        ImageSelector selector(image->intrinsics, ImageSelector::ModeAll);
        
        auto res = stitcher.Stitch(selector.GenerateDebugImages());
        imwrite("dbg/points.jpg", res->image);
        
        return 0;*/

        //Create stack-local ref to mat. Clear image mat.
        //This is to simulate hard memory management.
        Mat tmpMat = image->img;
        
        image->img = Mat(0, 0, CV_8UC3);
        
        image->dataRef.data = tmpMat.data;
        image->dataRef.width = tmpMat.cols;
        image->dataRef.height = tmpMat.rows;
        image->dataRef.colorSpace = colorspace::RGB;

        if(i == 0) {
            pipe = shared_ptr<Pipeline>(new Pipeline(Pipeline::iosBase, Pipeline::iosZero, image->intrinsics, RecorderGraph::ModeTruncated, true));
        }

        pipe->Push(image);
        tmpMat.release();
    }
    
    if(pipe->HasResults()) {
        {
            auto left = pipe->FinishLeft();
            imwrite("dbg/left.jpg", left->image);
            left->image.release();  
            left->mask.release();  
        }
        {
            auto right = pipe->FinishRight();
            imwrite("dbg/right.jpg", right->image);    
            right->image.release();  
            right->mask.release();  
        }
    } else {
        cout << "No results." << endl;
    }

    pipe->Dispose();

    return 0;
}
