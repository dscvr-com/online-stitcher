#include <iostream>
#include <algorithm>
#include <memory>

#include "pipeline.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

int IdFromFileName(const string &in) {
    size_t l = in.find_last_of("/");
    if(l == string::npos) {
        return ParseInt(in.substr(0, in.length() - 4));
    } else {
        return ParseInt(in.substr(l + 1, in.length() - 5 - l));
    }
}

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) > IdFromFileName(b);
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
        auto image = ImageFromFile(files[0]);
        
        if(i == 0) {
            pipe = shared_ptr<Pipeline>(new Pipeline(Pipeline::iosBase, image->intrinsics));
        }

        pipe->Push(image);
    }
    
    auto left = pipe->FinishLeft();
    auto right = pipe->FinishRight();
    
    imwrite("dbg/left.jpg", left->image);    
    imwrite("dbg/right.jpg", right->image);    

    return 0;
}
