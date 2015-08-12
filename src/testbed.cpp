#include <iostream>
#include <algorithm>

#include "image.hpp"
#include "visualAligner.hpp"
#include "simpleSphereStitcher.hpp"
#include "streamAligner.hpp"
#include "monoStitcher.hpp"
#include "io.hpp"
#include "wrapper.hpp"
#include "imageSelector.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

RStitcher stitcher;

bool CompareById (const ImageP a, const ImageP b) { return (a->id < b->id); }

void Stitch(vector<ImageP> imgs, string name = "stitched.jpg", bool debug = false) {
    StitchingResult* res = stitcher.Stitch(imgs, debug);

    imwrite("dbg/" + name, res->image);

    delete res;
}

void Align(vector<ImageP> imgs) {
    VisualAligner aligner;
    int n = imgs.size();

    for(int i = 0; i < n; i++) {
        aligner.FindKeyPoints(imgs[i]);
    }

    Mat last = Mat::eye(3, 3, CV_64F);
    imgs[0]->extrinsics = last;
    for(int i = 0; i < n - 1; i++) {
        MatchInfo* info = aligner.FindHomography(imgs[i], imgs[i + 1]);
        if(info->valid) {
                cout << endl;
                cout << "Homography between " << i << " and " << (i + 1) << ": " << info->homography << endl;

                //cout << "Rotation between " << i << " and " << (i + 1) << ": " << info->rotations << endl;
                //cout << "Sensor between " << i << " and " << (i + 1) << ": " << (GetDistanceY(imgs[i]->extrinsics, imgs[i+1]->extrinsics) * 180 / M_PI) << endl;
                //cout << "Translation between " << i << " and " << (i + 1) << ": " << info->translations << endl;
                cout << "Error between " << i << " and " << (i + 1) << ": " << info->error << endl;
                //TODO: Ist this correct? Check Mat type!
                imgs[i + 1]->extrinsics = info->rotations[0] * last;
                last = imgs[i + 1]->extrinsics;
                cout << "New Extrinsics: " << imgs[i + 1]->extrinsics  << endl;

        } else {
                imgs[i]->extrinsics = Mat::eye(4, 4, CV_64F);
                cout << "Matches between " << i << " and " << (i + 1) << " invalid" << endl;
        }
    }

    Stitch(imgs, "aligned.jpg");
}


vector<StereoImageP> Make3D(vector<ImageP> images) {
    vector<StereoImageP> stereos;
    int n = images.size();


    ImageSelector selector(images[0]->intrinsics);
    //Stitch(selector.GenerateDebugImages(), "dbd_select.jpg", true);

    ImageP prev = NULL;
    ImageP first = NULL;

    //Select good images and 3dify. Todo: Make optimal decisions. 
    //Handle missing images. Handle multi-rings. 
    for(int i = 0; i < n; i++) {
        if(selector.FitsModel(images[i])) {
            if(prev != NULL) {
                StereoImageP img = CreateStereo(prev, images[i]);
                if(img->valid) {
                    stereos.push_back(img);
                }
            }
            if(first == NULL)
                first = images[i];
            prev = images[i];
            //Todo - f & l
        }
    }

    //Wrap around end
    StereoImageP img = CreateStereo(prev, first);
    if(img->valid) {
        stereos.push_back(img);
    }

    return stereos;
}

void StreamAlign(vector<ImageP> images) {

    StreamAligner aligner;
    Stitch(images, "dbg_0_raw.jpg", true);

    cout << "RAW OUT FINISHED" << endl;

    //We need to do this if our 
    //pictures were recorded "upside-down".
    //The phone adjusts image orientation for us,
    //but the sensors are not adjusted. An incorrect
    //rotation results. 
    //TODO: Make this decision automatically. 
    //TODO: Might not be needed if code runs on phone with raw input 
    //data. 
    bool isLandscapeFlipped = false;
    bool isIos = true;

    if(isLandscapeFlipped) {
        double landscapeLtoRData[] = {-1, 0, 0, 0,
                                     0, -1, 0, 0, 
                                     0, 0, 1, 0,
                                     0, 0, 0, 1};
        Mat landscapeLtoR(4, 4, CV_64F, landscapeLtoRData);

        for(size_t i = 0; i < images.size(); i++) {
            images[i]->extrinsics = landscapeLtoR * images[i]->extrinsics * landscapeLtoR;
        }
    }
    if(isIos) {
        double iosToSData[] = {0, 1, 0, 0,
                                     1, 0, 0, 0, 
                                     0, 0, 1, 0,
                                     0, 0, 0, 1};
        Mat iosToS(4, 4, CV_64F, iosToSData);

        for(size_t i = 0; i < images.size(); i++) {
            images[i]->extrinsics = iosToS * images[i]->extrinsics * iosToS;
        }
    }


   // Stitch(images, "dbg_1_prepared.jpg", true);
    cout << "PREPARE FINISHED" << endl;

    for(size_t i = 0; i < images.size(); i++) {
        aligner.Push(images[i]);
        images[i]->extrinsics = aligner.GetCurrentRotation().clone();
    }

    //stitcher.PrepareMatrices(images);

   // Stitch(images, "dbg_2_aligned.jpg", false);
    cout << "ALIGN FINISHED" << endl;

    //Before stereofiying, make sure that images are sorted correctly!
    vector<StereoImageP> stereos = Make3D(images);
    //Also, take care! The images are deleted within the make 3D process. 
    images.clear();

    vector<ImageP> imagesLeft;
    vector<ImageP> imagesRight;

    for(size_t i = 0; i < stereos.size(); i += 1) {
        imagesLeft.push_back(stereos[i]->A);
        imagesRight.push_back(stereos[i]->B);
    }
    cout << "3D PROCESS FINISHED" << endl;

    Stitch(imagesLeft, "dbg_3_left.jpg", false);
    Stitch(imagesRight, "dbg_4_right.jpg", false);
    cout << "3D OUT FINISHED" << endl;
}


int main(int argc, char* argv[]) {

    int n = argc - 1;
    vector<ImageP> imgs(n);

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        imgs[i] = ImageFromFile(imageName);

        //cout << "Loaded " << imgs[i]->source << endl;
        //cout << "ID " << imgs[i]->id << endl;
        //cout << "Size " << imgs[i]->img.size() << endl;
        //cout << "Intrinsics " << imgs[i]->intrinsics << endl;
        //cout << "Extrinsics " << imgs[i]->extrinsics << endl;
    }
    sort(imgs.begin(), imgs.end(), CompareById);

    //Align(imgs);
    StreamAlign(imgs);

    for(int i = 0; i < n; i++) {
       //We're already freeing during 3dify to safe memory.    
       //delete imgs[i];
    }

    return 0;
}
