#include "core.hpp"
#include "alignment.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace optonaut;


bool CompareById (const Image* a, const Image* b) { return (a->id < b->id); }

int main(int argc, char* argv[]) {
	int n = argc - 1;
    vector<Image*> imgs(n);
    Aligner aligner;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        imgs[i] = ImageFromFile(imageName);

        cout << "Loaded " << imgs[i]->source << endl;
        cout << "ID " << imgs[i]->id << endl;
        cout << "Size " << imgs[i]->img.size() << endl;
        cout << "Intrinsics " << imgs[i]->intrinsics << endl;
        cout << "Extrinsics " << imgs[i]->extrinsics << endl;

        aligner.FindKeyPoints(imgs[i]);
    }

    sort(imgs.begin(), imgs.end(), CompareById);

    for(int i = 0; i < n - 1; i++) {
        MatchInfo* info = aligner.FindHomography(imgs[i], imgs[i + 1]);
        if(info->valid) {
                cout << endl;
                cout << "Homography between " << i << " and " << (i + 1) << ": " << info->homography << endl;
                cout << "Fundamental between " << i << " and " << (i + 1) << ": " << info->fundamental << endl;
                cout << "Essential between " << i << " and " << (i + 1) << ": " << info->essential << endl;
                cout << "Translation between " << i << " and " << (i + 1) << ": " << info->hshift << " px" << endl;
                cout << "Rotation between " << i << " and " << (i + 1) << ": " << (info->hrotation * 180 / M_PI) << " degrees" << endl;
                cout << "Error between " << i << " and " << (i + 1) << ": " << info->error << endl;
        } else {
            cout << "Matches between " << i << " and " << (i + 1) << " invalid" << endl;
        }
    }

    for(int i = 0; i < n; i++) {
    	delete imgs[i];
    }

    return 0;
}
