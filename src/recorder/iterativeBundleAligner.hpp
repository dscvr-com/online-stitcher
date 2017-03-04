#include <iostream>
#include <algorithm>
#include <memory>

#include "../stitcher/simpleSphereStitcher.hpp"
#include "../stitcher/simplePlaneStitcher.hpp"
#include "../recorder/alignmentGraph.hpp"
#include "../recorder/recorderGraph.hpp"

#include "../common/ringProcessor.hpp"

using namespace std;
using namespace cv;

#ifndef OPTONAUT_ITERATIVE_BUNDLE_ALIGNER_HEADER
#define OPTONAUT_ITERATIVE_BUNDLE_ALIGNER_HEADER

namespace optonaut {
class IterativeBundleAligner {
    private:
        static const bool drawDebug = false;

    public:
        static void DrawDebugGraph(const vector<InputImageP> &images,
                const RecorderGraph &graph, 
                const BiMap<size_t, uint32_t> &imagesToTargets,
                const AlignmentGraph::Edges &edges,
                std::string filename) {

            SimpleSphereStitcher debugger(200);
            auto res = debugger.Stitch(images);
            cv::Point imgCenter = res->corner;
            std::map<size_t, InputImageP> imageById;

            for(auto img : images) {
                imageById[img->id] = img;
            }

            for(auto edge : edges) {
               if(edge.value.valid) {
                   InputImageP a = imageById[edge.from]; 
                   InputImageP b = imageById[edge.to]; 

                   uint32_t pidA = 0, pidB = 0;
                   Assert(imagesToTargets.GetValue(a->id, pidA));
                   Assert(imagesToTargets.GetValue(b->id, pidB));
                   SelectionPoint tA, tB;
                   Assert(graph.GetPointById(pidA, tA));
                   Assert(graph.GetPointById(pidB, tB));

                   if(tA.ringId != 1 && tA.ringId != tB.ringId)
                       continue; //Only draw cross-lines if origin at ring 0

                   cv::Point aCenter = debugger.WarpPoint(a->intrinsics,
                           a->adjustedExtrinsics, 
                           a->image.size(), cv::Point(0, 0)) - imgCenter;
                   cv::Point bCenter = debugger.WarpPoint(b->intrinsics,
                           b->adjustedExtrinsics, 
                           b->image.size(), cv::Point(0, 0)) - imgCenter;

                   //Make sure a is always left. 
                   if(aCenter.x > bCenter.x) {
                        swap(aCenter, bCenter);
                   }

                   double dPhi = edge.value.dphi * 10;
                   //double dTheta = edge.value.dtheta * 10;

                   if(dPhi != dPhi)
                       continue;

                   Scalar color(255 * min(1.0, max(0.0, -dPhi)), 
                               0, 
                               255 * min(1.0, max(0.0, dPhi)));
                   
                   int thickness = 6;

                   if(edge.value.quartil) {
                        thickness = 2;
                   }

                   if(bCenter.x - aCenter.x > res->image.cols / 2) {
                        cv::line(res->image.data, 
                           aCenter, bCenter - cv::Point(res->image.cols, 0),
                           color, thickness);
                        cv::line(res->image.data, 
                           aCenter + cv::Point(res->image.cols, 0), bCenter,
                           color, thickness);
                   } else {
                        cv::line(res->image.data, 
                           aCenter, bCenter, color, thickness);
                   }
                }
            }

            // TODO: add target ID again. 
            DrawPointsOnPanorama(res->image.data, 
                    ExtractExtrinsics(images), images[0]->intrinsics, 
                    images[0]->image.size(), 800, res->corner);

            imwrite(filename, res->image.data);
        }

        IterativeBundleAligner() { }
        double focalLenAdjustment;


        double GetFocalLenAdjustment() {
            return focalLenAdjustment;
        }   

        void Align(const vector<InputImageP> &images, 
                const RecorderGraph &graph, 
                const BiMap<size_t, uint32_t> &imagesToTargets,
                const int roundTresh = 15, const double errorTresh = 10) {

            SimpleSphereStitcher debugger(200);

            AssertFalseInProduction(drawDebug);
            
            if(drawDebug) {
                auto res = debugger.Stitch(images);
                imwrite("dbg/iterative_bundler_input.jpg", res->image.data);
            }

            int n = (int)images.size();
            
            for(int k = 0; k < roundTresh; k++) {
                AlignmentGraph aligner;
                int matches = 0, outliers = 0, forced = 0, noOverlap = 0;
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < i; j++) {
                        auto corr = aligner.Register(images[i], images[j]);

                        if(corr.valid) {
                           matches++; 
                        } else if(corr.rejectionReason == 
                                PairwiseCorrelator::RejectionOutOfWindow ||
                                corr.rejectionReason == 
                                PairwiseCorrelator::RejectionDeviationTest) {
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

                auto rings = graph.SplitIntoRings(images);

                deque<double> adjustments;

                for(size_t i = 0; i < rings.size(); i++) {
                    auto &ring = rings[i];
                    double dist = 0;
                    auto addDist = [&dist, &aligner] 
                        (const InputImageP a, const InputImageP b) {
                        AlignmentDiff diff;
                        AlignmentGraph::Edge edge(0, 0, diff);

                        if(aligner.GetEdge(a->id, b->id, edge)) {
                            dist += edge.value.dphi;
                        }
                    };

                    RingProcessor<InputImageP> 
                        sum(1, addDist, [](const InputImageP&) {});

                    sum.Process(ring);

                    cout << "Ring dist sum: " << dist << endl;
                    double focalLenAdjustment = (1 - dist / (M_PI * 2));
                    cout << "Estimated focal length adjustment factor: " 
                        << focalLenAdjustment << endl;

                    adjustments.push_back(focalLenAdjustment);
                }

                focalLenAdjustment = Mean(adjustments);
                cout << " Final focal length adjustment factor: " 
                    << focalLenAdjustment << endl;

                for(auto img : images) {
                   img->intrinsics.at<double>(0, 0) *= focalLenAdjustment; 
                   img->intrinsics.at<double>(1, 1) *= focalLenAdjustment; 
                }

                //Needed for iteration
                for(auto img : images) {
                    aligner.Apply(img);
                    img->adjustedExtrinsics.copyTo(img->originalExtrinsics);
                }
                
                if(drawDebug) { 
                    DrawDebugGraph(images, graph, imagesToTargets, edges, "dbg/aligned_iteration_" + ToString(k) + ".jpg");
                }

                if(outError < errorTresh) {
                    break;
                }
            }
        }
};
}

#endif
