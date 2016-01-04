#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "common/functional.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "recorder/alignmentGraph.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "stitcher/multiRingStitcher.hpp"
#include "minimal/stereoConverter.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    bool drawWeights = true;
    bool drawDebug = true;
    bool outputUnalignedStereo = false;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);

    vector<InputImageP> allImages;
    SimpleSphereStitcher debugger;

    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();
    map<size_t, InputImageP> imageById;

    
    for(int i = 0; i < n; i += 1) {
        auto img = InputImageFromFile(files[i], true); 
        img->originalExtrinsics = base * zero * 
            img->originalExtrinsics.inv() * baseInv;
        img->adjustedExtrinsics = img->originalExtrinsics;
        imageById[img->id] = img;

        allImages.push_back(img);
    }
    
    RecorderGraph fullGraph = RecorderGraphGenerator::Generate(
            allImages[0]->intrinsics, 
            RecorderGraph::ModeTruncated, 
            2, 0, 4);

    BiMap<uint32_t, uint32_t> fullToHalf; 
    BiMap<size_t, uint32_t> fullImagesToTargets; 
    BiMap<size_t, uint32_t> imagesToTargets; 
    auto fullImages = fullGraph.SelectBestMatches(allImages, fullImagesToTargets);
    vector<InputImageP> halfImages;

    RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(
            fullGraph, 
            fullImagesToTargets,
            imagesToTargets,
            fullToHalf, 4);

    //Iterate over imagesToTArgets and fill halfImages
    
    for(auto pair : imagesToTargets) {
        halfImages.push_back(imageById.at(pair.first));
    }

    n = halfImages.size();
    
    cout << "Selecting " << n << " images for further processing." << endl;

    for(auto img : halfImages) {
        Assert(!img->IsLoaded());
        img->image.Load();
        pyrDown(img->image.data, img->image.data);
        pyrDown(img->image.data, img->image.data);
        img->image = Image(img->image.data);
    }

    std::pair<StitchingResultP, StitchingResultP> stereoRes; 
    
    if(outputUnalignedStereo) {
        stereoRes = minimal::StereoConverter::Stitch(halfImages, halfGraph);

        imwrite("dbg/stereo_una_left.jpg", stereoRes.first->image.data);
        imwrite("dbg/stereo_una_right.jpg", stereoRes.second->image.data);
    }

    if(drawDebug) {
        auto res = debugger.Stitch(halfImages);
        imwrite("dbg/aa_input.jpg", res->image.data);
    }
    for(int k = 0; k < 10; k++) {
        AlignmentGraph aligner(halfGraph, imagesToTargets);
        int matches = 0, outliers = 0, forced = 0, noOverlap = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < i; j++) {
                auto corr = aligner.Register(halfImages[i], halfImages[j]);

                if(corr.valid) {
                   matches++; 
                } else if(corr.rejectionReason == 
                        PairwiseCorrelator::RejectionOutOfWindow ||
                        corr.rejectionReason == 
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
       
        if(drawDebug) { 
            auto res = debugger.Stitch(halfImages);
            Point imgCenter = res->corner;


            if(drawWeights) { 
                for(auto edge : edges) {
                   if(edge.value.valid) {
                       InputImageP a = imageById[edge.from]; 
                       InputImageP b = imageById[edge.to]; 

                       uint32_t pidA, pidB;
                       Assert(imagesToTargets.GetValue(a->id, pidA));
                       Assert(imagesToTargets.GetValue(b->id, pidB));
                       SelectionPoint tA, tB;
                       Assert(halfGraph.GetPointById(pidA, tA));
                       Assert(halfGraph.GetPointById(pidB, tB));

                       if(tA.ringId != 1 && tA.ringId != tB.ringId)
                           continue; //Only draw cross-lines if origin at ring 0

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

                       //if(!edge.value.forced) {
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
                       //}
                    }
                }

                for(auto img : halfImages) {

                    uint32_t targetId;

                    Assert(imagesToTargets.GetValue(img->id, targetId));

                    Point center = debugger.WarpPoint(img->intrinsics, 
                               img->adjustedExtrinsics, 
                               img->image.size(), Point(0, 0)) - imgCenter;

                    cv::circle(res->image.data, center, 8, 
                        Scalar(0xc0, 0xc0, 0x00), -1);

                    Point offset(-20, -20);

                    if(targetId % 2 == 0)
                        offset.y -= 50;
                    
                    cv::putText(res->image.data, ToString(targetId), center + offset, FONT_HERSHEY_PLAIN, 3, Scalar(0x00, 0xFF, 0x00), 3);
                }
            }
            imwrite("dbg/aligned_" + ToString(k) + ".jpg", res->image.data);
        }
        
        for(auto img : halfImages) {
            aligner.Apply(img);
            img->adjustedExtrinsics.copyTo(img->originalExtrinsics);
        }
    }
   
   //Todo: Remove all unused parameters.  
    auto adjustedImages = RecorderGraphGenerator::AdjustFromSparse(
            halfImages, 
            halfGraph , 
            imagesToTargets,
            allImages,
            fullGraph, 
            fullImagesToTargets,
            fullToHalf);
    
    auto finalImages = fullGraph.SelectBestMatches(adjustedImages, fullImagesToTargets);
    
    for(auto img : finalImages) {
        if(!img->IsLoaded()) {
            //Load unloaded images - take car about the size. 
            img->image.Load();
            pyrDown(img->image.data, img->image.data);
            pyrDown(img->image.data, img->image.data);
            img->image = Image(img->image.data);
        }
    }

    stereoRes = minimal::StereoConverter::Stitch(finalImages, fullGraph);

    imwrite("dbg/stereo_left.jpg", stereoRes.first->image.data);
    imwrite("dbg/stereo_right.jpg", stereoRes.second->image.data);
        
    return 0;
}
