#include <iostream>
#include <algorithm>
#include <memory>

#include "../stitcher/simpleSphereStitcher.hpp"
#include "../stitcher/simplePlaneStitcher.hpp"
#include "../recorder/alignmentGraph.hpp"
#include "../recorder/recorderGraph.hpp"

using namespace std;
using namespace cv;

namespace optonaut {
class IterativeBundleAligner {
    private:
        static const bool drawDebug = false;
        static const bool drawWeights = false;

        static void DrawDebugGraph(const vector<InputImageP> &images,
                const RecorderGraph &graph, 
                const BiMap<size_t, uint32_t> &imagesToTargets,
                const AlignmentGraph::Edges &edges,
                int k) {

            SimpleSphereStitcher debugger;
            auto res = debugger.Stitch(images);
            Point imgCenter = res->corner;
            std::map<size_t, InputImageP> imageById;

            for(auto img : images) {
                imageById[img->id] = img;
            }

            if(drawWeights) { 
                for(auto edge : edges) {
                   if(edge.value.valid) {
                       InputImageP a = imageById[edge.from]; 
                       InputImageP b = imageById[edge.to]; 

                       uint32_t pidA, pidB;
                       Assert(imagesToTargets.GetValue(a->id, pidA));
                       Assert(imagesToTargets.GetValue(b->id, pidB));
                       SelectionPoint tA, tB;
                       Assert(graph.GetPointById(pidA, tA));
                       Assert(graph.GetPointById(pidB, tB));

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
                    }
                }

                for(auto img : images) {

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
                    
                    cv::putText(res->image.data, ToString(targetId), 
                            center + offset, FONT_HERSHEY_PLAIN, 3, 
                            Scalar(0x00, 0xFF, 0x00), 3);
                }
            }
            imwrite("dbg/aligned_" + ToString(k) + ".jpg", res->image.data);
        }

    public: 
        static void Align(const vector<InputImageP> &images, 
                const RecorderGraph &graph, 
                const BiMap<size_t, uint32_t> &imagesToTargets,
                const int roundTresh = 15, const double errorTresh = 10) {

            AssertFalseInProduction(drawDebug);
            AssertFalseInProduction(drawWeights);
            
            if(drawDebug) {
                SimpleSphereStitcher debugger;
                auto res = debugger.Stitch(images);
                imwrite("dbg/iterative_bundler_input.jpg", res->image.data);
            }

            int n = (int)images.size();
            
            for(int k = 0; k < roundTresh; k++) {
                AlignmentGraph aligner(graph, imagesToTargets);
                int matches = 0, outliers = 0, forced = 0, noOverlap = 0;
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < i; j++) {
                        auto corr = aligner.Register(images[i], images[j]);

                        if(corr.valid) {
                           matches++; 
                        } else if(corr.rejectionReason == 
                                PairwiseCorrelator::RejectionOutOfWindow ||
                                corr.rejectionReason == 
                                PairwiseCorrelator::RejectionInverseTest) {
                           outliers++; 
                        } else if(corr.rejectionReason == 
                                PairwiseCorrelator::RejectionNoOverlap) {
                           noOverlap++; 
                        }
                        if(corr.forced) {
                            forced++;
                        }
                    }
                }

                double outError = 0;

                AlignmentGraph::Edges edges = aligner.FindAlignment(outError);

                cout << "Pass " << k << ", error: " << outError << ", matches: " 
                    << matches << " (real: " << (matches - forced) << ")" << 
                    ", outliers: " << outliers << ", no overlap: " << noOverlap << 
                    ", forced: " << forced<< endl;

                if(drawDebug) { 
                    DrawDebugGraph(images, graph, imagesToTargets, edges, k);
                }
                
                //Needed for iteration
                for(auto img : images) {
                    aligner.Apply(img);
                    img->adjustedExtrinsics.copyTo(img->originalExtrinsics);
                }

                if(outError < errorTresh) {
                    break;
                }
            }
        }
};
}
