#include "core.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace optonaut;

int main(int argc, char* argv[]) {
	vector<Image*> imgs(argc - 1);

    for(int i = 0; i < argc - 1; i++) {
        string imageName(argv[i + 1]);
        imgs[i] = ImageFromFile(imageName);

        cout << "Loaded " << imgs[i]->source << endl;
        cout << "ID " << imgs[i]->id << endl;
        cout << "Size " << imgs[i]->img.size() << endl;
        cout << "Intrinsics " << imgs[i]->intrinsics << endl;
        cout << "Extrinsics " << imgs[i]->extrinsics << endl;
    }

    for(int i = 0; i < argc - 1; i++) {
    	delete imgs[i];
    }

    return 0;
}
