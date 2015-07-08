#include "core.hpp"
#include "alignment.hpp"
#include "simpleSphereStitcher.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace optonaut;


bool CompareById (const Image* a, const Image* b) { return (a->id < b->id); }

void Align(vector<Image*> imgs) {
    Aligner aligner;
    int n = imgs.size();

    sort(imgs.begin(), imgs.end(), CompareById);

    for(int i = 0; i < n; i++) {
        aligner.FindKeyPoints(imgs[i]);
    }

    for(int i = 0; i < n - 1; i++) {
        MatchInfo* info = aligner.FindHomography(imgs[i], imgs[i + 1]);
        if(info->valid) {
                cout << endl;
                cout << "Homography between " << i << " and " << (i + 1) << ": " << info->homography << endl;

                cout << "Rotation between " << i << " and " << (i + 1) << ": " << (info->hrotation * 180 / M_PI) << endl;
                cout << "Sensor between " << i << " and " << (i + 1) << ": " << (GetDistanceY(imgs[i]->extrinsics, imgs[i+1]->extrinsics) * 180 / M_PI) << endl;
                cout << "Shift between " << i << " and " << (i + 1) << ": " << info->hshift << endl;
                cout << "Error between " << i << " and " << (i + 1) << ": " << info->error << endl;


        } else {
            cout << "Matches between " << i << " and " << (i + 1) << " invalid" << endl;
        }
    }
}

void Stitch(vector<Image*> imgs) {
    RStitcher stitcher;
    stitcher.PrepareMatrices(imgs);
    StitchingResult* res = stitcher.Stitch(imgs);

    imwrite("dbg/stitched.jpg", res->image);

    delete res;
}

int main(int argc, char* argv[]) {
    int n = argc - 1;
    vector<Image*> imgs(n);

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        imgs[i] = ImageFromFile(imageName);

        cout << "Loaded " << imgs[i]->source << endl;
        cout << "ID " << imgs[i]->id << endl;
        cout << "Size " << imgs[i]->img.size() << endl;
        cout << "Intrinsics " << imgs[i]->intrinsics << endl;
        cout << "Extrinsics " << imgs[i]->extrinsics << endl;
    }

    Stitch(imgs);

    for(int i = 0; i < n; i++) {
        delete imgs[i];
    }

    return 0;
}