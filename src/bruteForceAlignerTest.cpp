#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "recorder/alignmentGraph.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);

    vector<InputImageP> images;
    SimpleSphereStitcher debugger;

    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();
    
    for(int i = 0; i < n; i += 1) {
        auto img = InputImageFromFile(files[i], true); 
        img->originalExtrinsics = base * zero * 
            img->originalExtrinsics.inv() * baseInv;
        img->adjustedExtrinsics = img->originalExtrinsics;

        images.push_back(img);
    }

    RecorderGraphGenerator gen;
    RecorderGraph graph = gen.Generate(images[0]->intrinsics, RecorderGraph::ModeTruncated, 0.5, 0);

    images = graph.SelectBestMatches(images);
    n = images.size();
    
    cout << "Selecting " << n << " images for further processing." << endl;

    map<size_t, InputImageP> imageById;

    for(auto img : images) {
        img->image.Load();
        pyrDown(img->image.data, img->image.data);
        pyrDown(img->image.data, img->image.data);
        img->image = Image(img->image.data);

        imageById[img->id] = img;
    }

    auto res = debugger.Stitch(images);
    imwrite("dbg/aa_input.jpg", res->image.data);

    for(int k = 0; k < 250; k++) {
        AlignmentGraph aligner;
        int matches = 0, outliers = 0, forced = 0, noOverlap = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < i; j++) {
                auto corr = aligner.Register(images[i], images[j]);
                if(corr.valid) {
                   matches++; 
                } else if(corr.rejectionReason == 
                        PairwiseCorrelator::RejectionInverseTest) {
                   outliers++; 
                   //cout << "Outlier: " << images[i]->id << " <-> " << images[j]->id << ": " << corr.error << endl;
                } else if(corr.rejectionReason == 
                        PairwiseCorrelator::RejectionNoOverlap) {
                   noOverlap++; 
                }
                if(corr.forced) {
                    forced++;
                }
            }
        }

        AlignmentGraph::Edges edges = aligner.FindAlignment();

        cout << "Pass " << k << ", matches: " << matches << 
            " (real: " << (matches - forced) << ")" << 
            ", outliers: " << outliers << ", no overlap: " << noOverlap << 
            ", forced: " << forced<< endl;
        
        res = debugger.Stitch(images);
        for(auto edge : edges) {
           if(edge.value.valid) {
               InputImageP a = imageById[edge.from]; 
               InputImageP b = imageById[edge.to]; 

               if(a->ringId != 1 && a->ringId != b->ringId)
                   continue; //Only draw cross-lines if origin at ring 0

               Point imgCenter = res->corner;

               Point aCenter = debugger.WarpPoint(a->intrinsics, 
                       a->adjustedExtrinsics, 
                       a->image.size(), Point(0, 0)) - imgCenter;
               Point bCenter = debugger.WarpPoint(b->intrinsics, 
                       b->adjustedExtrinsics, 
                       b->image.size(), Point(0, 0)) - imgCenter;

               //Make sure a is always left. 
               if(aCenter.x > bCenter.x) {
                    swap(aCenter, bCenter);
               }

               double dPhi = edge.value.dphi * 10;

               Scalar color(255 * min(1.0, max(0.0, -dPhi)), 
                           0, 
                           255 * min(1.0, max(0.0, dPhi)));
               
               int thickness = 6;

               if(edge.value.quartil) {
                    thickness = 2;
               }

               if(edge.value.forced) {
                   continue;
                   // color = Scalar(0xc0, 0xc0, 0xc0);
                   // thickness = 2;
               }
               

               if(bCenter.x - aCenter.x > res->image.cols / 2) {
                    cv::line(res->image.data, 
                       aCenter, bCenter - Point(res->image.cols, 0), 
                       color, thickness);
                    cv::line(res->image.data, 
                       aCenter + Point(res->image.cols, 0), bCenter,
                       color, thickness);
               } else {
                    cv::line(res->image.data, 
                       aCenter, bCenter, color, thickness);
               }

               cv::circle(res->image.data, aCenter, 8, Scalar(0xc0, 0xc0, 0x00), -1);
            }
        }
        imwrite("dbg/aligned_" + ToString(k) + ".jpg", res->image.data);
        
        for(auto img : images) {
            aligner.Apply(img);
            img->adjustedExtrinsics.copyTo(img->originalExtrinsics);
        }
        
    }
        
    return 0;
}
