#include "core.hpp"
#include "VisualAligner.hpp"
#include "simpleSphereStitcher.hpp"
#include "streamAligner.hpp"
#include "monoStitcher.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace optonaut;

RStitcher stitcher;

bool CompareById (const Image* a, const Image* b) { return (a->id < b->id); }

void Stitch(vector<Image*> imgs, string name = "stitched.jpg", bool debug = false) {
    StitchingResult* res = stitcher.Stitch(imgs, debug);

    imwrite("dbg/" + name, res->image);

    delete res;
}

void Align(vector<Image*> imgs) {
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


vector<StereoImage*> Make3D(vector<Image*> images) {
    vector<StereoImage*> stereos;
    int n = images.size();

    for(int i = 0; i < n; i++) {
        StereoImage* img = CreateStereo(images[i], images[(i + 1) % n]);
        if(img->valid) {
            stereos.push_back(img);
        }
    }

    return stereos;
}


void StreamAlign(vector<Image*> images) {
    StreamAligner aligner;
    Stitch(images, "dbg_0_raw.jpg", false);
    stitcher.PrepareMatrices(images);

    Stitch(images, "dbg_1_prepared.jpg", false);

    for(size_t i = 0; i < images.size(); i++) {
        aligner.Push(images[i]);
        images[i]->extrinsics = aligner.GetCurrentRotation().clone();
    }

    Stitch(images, "dbg_2_aligned.jpg", false);

    vector<StereoImage*> stereos = Make3D(images);

    vector<Image*> imagesLeft;
    vector<Image*> imagesRight;

    for(size_t i = 0; i < stereos.size(); i++) {
        imagesLeft.push_back(&(stereos[i]->A));
        imagesRight.push_back(&(stereos[i]->B));
    }

    Stitch(imagesLeft, "dbg_3_left.jpg");
    Stitch(imagesRight, "dbg_4_right.jpg");
}

int main(int argc, char* argv[]) {
    int n = argc - 1;
    vector<Image*> imgs(n);

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
        delete imgs[i];
    }

    return 0;
}