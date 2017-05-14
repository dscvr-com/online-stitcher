#include "../math/stat.hpp"
#include "../recorder/alignmentGraph.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"
#include "../common/static_timer.hpp"

#ifndef OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_HEADER
#define OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_HEADER

namespace optonaut {
class ImageCorrespondenceFinder : public SelectionSink {
    private: 
        std::deque<SelectionInfo> largeImages;
        std::vector<SelectionInfo> miniImages;

        cv::Ptr<cv::WarperCreator> warperFactory;
        cv::Ptr<cv::detail::RotationWarper> warper;

        AlignmentGraph alignment;
        ExposureCompensator exposure;
        PairwiseCorrelator matcher;

        Sink<std::vector<InputImageP>> &outSink;

        typedef std::map<std::pair<size_t, size_t>, Point2d> PlanarOffsetList;
        PlanarOffsetList planarCorrelations;

        const RecorderGraph &graph;

        const int downsample = 0;
        const bool debug = false;
        bool focalLenAdjustmentOn;
        bool fullAlignmentOn;
        bool ringClosingOn;

        void ComputeMatch(const SelectionInfo &a, const SelectionInfo &b, 
                          int overlapArea) {
            STimer timer;
            //int minSize = min(a.image->image.cols, b.image->image.rows) / 3;
            auto res = matcher.Match(a.image, b.image, 4, 4, false, 0.5, 1.8);

            timer.Tick("Compute match");

            Log << "B adj extrinsics: " << b.image->adjustedExtrinsics;
            Log << "A adj extrinsics: " << a.image->adjustedExtrinsics;
            Log << "B orig extrinsics: " << b.image->originalExtrinsics;
            Log << "A orig extrinsics: " << a.image->originalExtrinsics;

            Log << "Computing match between " << a.image->id << " and " << b.image->id << ": " << res.valid;

            // TODO - Something is broken when correlating the images!

            if(res.valid) {
                planarCorrelations.emplace(std::make_pair(a.image->id, b.image->id), res.angularOffset);
                planarCorrelations.emplace(std::make_pair(b.image->id, a.image->id), -res.angularOffset);
                {
                    AlignmentDiff aToB, bToA;

                    aToB.dphi = -res.angularOffset.y;
                    if(a.closestPoint.ringId == b.closestPoint.ringId) {
                        aToB.dtheta = res.angularOffset.x;
                    } else {
                        aToB.dtheta = NAN; 
                        // Only work with vertical offsets for neighbors. 
                    }
                    aToB.dx = res.offset.x * std::pow(2, downsample);
                    aToB.dy = res.offset.y * std::pow(2, downsample);
                    aToB.overlap = 1; // res.correlationCoefficient * 2;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;

                    Log << "Offset " << a.image->id << 
                        " <> " << b.image->id << 
                        " : " << res.offset * std::pow(2, downsample) << 
                        ", angular: " << res.angularOffset;
                    
                    bToA = aToB;
                    bToA.dphi *= -1;
                    bToA.dtheta *= -1;
                    bToA.dx *= -1;
                    bToA.dy *= -1;

                    alignment.InsertCorrespondence(a.image->id, b.image->id, 
                            aToB, bToA);
                }
                {
                    ExposureDiff aToB, bToA;
                    aToB.n = overlapArea;
                    aToB.iFrom = res.gainA;
                    aToB.iTo = res.gainB;

                    Log << "Gains: " << res.gainA << "/" << res.gainB;
                    
                    bToA.n = overlapArea;
                    bToA.iFrom = res.gainB;
                    bToA.iTo = res.gainA;
                        
                    exposure.InsertCorrespondence(
                            a.image->id, b.image->id, aToB, bToA);
                }
            } else {
                Log << "Skipping" << a.image->id << 
                    " <> " << b.image->id;
            }
            timer.Tick("Added to alignment graph");
        }

        PlanarOffsetList CorrespondenceCrossProduct(PlanarOffsetList correspondences) {
            PlanarOffsetList res;
            for(auto a : correspondences) {
                res.emplace(a);
                for(auto b : correspondences) {
                    res.emplace(std::make_pair(a.first.first, b.first.second), a.second + b.second);
                }
            }
            return res;
        }
    public:
        ImageCorrespondenceFinder(
            Sink<std::vector<InputImageP>> &outSink, 
            const RecorderGraph &fullGraph, 
            bool focalLenAdjOn = true, 
            bool fullAlignmentOn = true,
            bool ringClosingOn = true) :
        outSink(outSink), graph(fullGraph),
        focalLenAdjustmentOn(focalLenAdjOn), 
        fullAlignmentOn(fullAlignmentOn),
        ringClosingOn(ringClosingOn) { 
                warperFactory = new cv::SphericalWarper();
                warper = warperFactory->create(static_cast<float>(1600));
                AssertFalseInProduction(debug);
        }

        virtual void Push(SelectionInfo info) {
            Log << "Received Image: " << info.image->id;
            Log << "K: " << info.image->intrinsics;

            // Downsample the iamge - create a minified copy. 
            auto miniCopy = std::make_shared<InputImage>(*(info.image));

            cv::Mat small; 
   
            if(downsample >= 1) {
                pyrDown(miniCopy->image.data, small);
                for(int i = 1; i < downsample; i++) {
                    pyrDown(small, small);
                }              
                miniCopy->image = Image(small); 
            }

            
            SelectionInfo infoCopy = info;
            infoCopy.image = miniCopy;
            // Now match with all possible images.
            
            for(auto cand : miniImages) {
                auto roiCand = GetOuterRectangle(*warper, cand.image);
                auto inCand = GetOuterRectangle(*warper, infoCopy.image);

                int overlapArea = (roiCand & inCand).area();

                bool sameRing = cand.closestPoint.ringId == infoCopy.closestPoint.ringId;


            // TODO - check if the last step 
            // calcs the correspondence in the correc direction!
                bool neighbors = sameRing && (
                        std::abs((int)cand.closestPoint.localId - (int)infoCopy.closestPoint.localId) == 1 ||
                        (size_t)std::abs((int)cand.closestPoint.localId - (int)infoCopy.closestPoint.localId) == graph.GetRings()[cand.closestPoint.ringId].size() - 1);

                //Log << "NEIGH, Area: " << overlapArea << ", " << neighbors;


                if(neighbors) {
                //if(overlapArea > inCand.area() / 5.0f * 4.0f) {
                    //Log << "Points " << cand.closestPoint.localId << " and " << infoCopy.closestPoint.localId;
                    ComputeMatch(infoCopy, cand, overlapArea);
                }
            } 
            
            miniImages.push_back(infoCopy);
            largeImages.push_back(info);
        }

        const std::map<std::pair<size_t, size_t>, cv::Point2d>& GetPlanarOffsets() const {
            return planarCorrelations;
        }

        virtual void Finish() {

            if(largeImages.size() == 0)  {
                outSink.Finish();
                return;
            }

            double error = 0; 

            miniImages.clear();
            vector<InputImageP> images;
            for(auto info : largeImages) {
                images.push_back(info.image);
            }
            auto rings = graph.SplitIntoRings(images); 
            deque<double> adjustments; 

            int currentRing = rings.size() / 2;

            if(debug) {
                auto images = fun::map<SelectionInfo, InputImageP>(largeImages, [](const SelectionInfo& x) -> InputImageP { return x.image; });
                SimpleSphereStitcher::StitchAndWrite("dbg/alignment_1_before.jpg", images);
                BiMap<size_t, uint32_t> imagesToTargets;
                std::vector<AlignmentGraph::Edge> edges;

                for(auto pair : alignment.GetEdges()) {
                    edges.insert(edges.end(), pair.second.begin(), pair.second.end());
                }

                for(auto info : largeImages) {
                    imagesToTargets.Insert(info.image->id, info.closestPoint.globalId);
                }

                // TODO - for debugging anyawy
                //IterativeBundleAligner::DrawDebugGraph(images, graph, imagesToTargets, edges, "dbg/alignment_0_weights.jpg");
            }

            // First, close all rings.
            // Do so in recording order
            while(currentRing >= 0 && currentRing < (int)rings.size() && ringClosingOn) {

                // What's missing: Matching between rings. Proper adjustment of whole graph. 
                AssertEQM(rings.size(), (size_t)1, "Only single ring supported at the moment");

                const vector<InputImageP> &ring = rings[currentRing];
                
                //1) Get first/last
                //is this true? 
                auto first = ring.front();
                auto last = ring.back();
                
                Log << "Closing ring between: " << first->id << " and " << last->id;

                //2) Use alignment data  
                auto edges = alignment.GetEdges(first->id, last->id);

                // Only do ring alignment if ring closing data is available. 
                if(edges.size() == (size_t)1) {
                    double angleAdjustment = edges[0]->value.dphi;

                    size_t n = ring.size();

                    //3) Align all images accordingly
                    //4) Don't forget to adjust the graph too!
                    for(size_t i = 0; i < n; i++) {
                        double ydiff = angleAdjustment * ((double)i / (double)n);
                        Mat correction;
                        CreateRotationY(ydiff, correction);
                        ring[i]->adjustedExtrinsics = correction * ring[i]->adjustedExtrinsics;

                        Log << "Adjusting " << ring[i]->id << " by " << ydiff;

                        vector<AlignmentGraph::Edge*> bwEdges;
/*
                        for(auto &fwdEdge : alignment.GetEdges()[ring[i]->id]) {
                            fwdEdge.value.dphi -= ydiff;
                            auto _bwEdges = alignment.GetEdges(fwdEdge.to, fwdEdge.from);
                            bwEdges.insert(bwEdges.begin(), _bwEdges.begin(), _bwEdges.end());
                        }

                        std::sort(bwEdges.begin(), bwEdges.end());
                        bwEdges.erase(unique(bwEdges.begin(), bwEdges.end()), bwEdges.end());
                        for(auto bwEdge : bwEdges) {
                            bwEdge->value.dphi += ydiff;
                        }
                        */
                    }

                    //4) TODO: correct graph for other rings. Alignment is probably sane, but
                    //it's way more efficient to "pre-turn" the rings. 
                    //DO NOT correct focal len
                }

                currentRing = graph.GetNextRing(currentRing);
            }
            
            if(debug) {
                //SimpleSphereStitcher::StitchAndWrite("dbg/alignment_ring_closed.jpg", largeImages);
                SimpleSphereStitcher::StitchAndWrite("dbg/alignment_2_ring_closed.jpg", fun::map<SelectionInfo, InputImageP>(largeImages, [](const SelectionInfo& x) -> InputImageP { return x.image; }));
            }

            // Second, perform global optimization. 
            alignment.FindAlignment(error); 
            exposure.FindGains();

            // Third, apply focal length corrections.  
            if(focalLenAdjustmentOn) {
                for(size_t i = 0; i < rings.size(); i++) {
                    auto ring = rings[i];

                    if(ring.size() == 0)
                        continue;

                    double dist = 0;

                    auto addDist = [&] 
                        (const InputImageP a, const InputImageP b) {
                        AlignmentDiff diff;
                        AlignmentGraph::Edge edge(0, 0, diff);

                        if(alignment.GetEdge(a->id, b->id, edge)) {
                            dist += edge.value.dphi;
                        }
                    };

                    RingProcessor<InputImageP> 
                        sum(1, addDist, [](const InputImageP&) {});

                    Log << "Ring alignment dist: " << dist;

                    sum.Process(ring);

                    double focalLenAdjustment = 1 / (1 - dist / (M_PI * 2));
                    Log << "Estimated focal length adjustment factor for ring "
                        << i << ": " 
                        << focalLenAdjustment;

                    adjustments.push_back(focalLenAdjustment);
                }

                double focalLenAdjustment = Mean(adjustments);
                Log << "Global focal length adjustment: "
                    << focalLenAdjustment;

                for(auto info: largeImages) {
                    Log << "Adjusting focal len for image " << info.image->id;
                    Log << "old k" << info.image->intrinsics;
                    info.image->intrinsics.at<double>(0, 0) *= focalLenAdjustment;
                    info.image->intrinsics.at<double>(1, 1) *= focalLenAdjustment;
                    Log << "New k" << info.image->intrinsics;
                }
            }
            
            if(debug) {
                SimpleSphereStitcher::StitchAndWrite("dbg/alignment_3_focal_len.jpg", fun::map<SelectionInfo, InputImageP>(largeImages, [](const SelectionInfo& x) -> InputImageP { return x.image; }));
            }

            if(fullAlignmentOn) {
                // Todo - not sure if apply is good here. 
                for(auto info : largeImages) {
                    Mat tmp(4, 4, CV_64F);
                    info.image->adjustedExtrinsics.copyTo(tmp);
                    alignment.Apply(info.image);
                    info.image->adjustedExtrinsics.copyTo(info.image->originalExtrinsics);
                    Log << "Extrinsic change: " << (tmp - info.image->adjustedExtrinsics);
                    // Exposure disabled for now. 
                    // exposure.Apply(info.image->image.data, info.image->id);
                }
            }
            
            if(debug) {
                //SimpleSphereStitcher::StitchAndWrite("dbg/alignment_adjusted.jpg", largeImages);
                SimpleSphereStitcher::StitchAndWrite("dbg/alignment_4_aligned.jpg", fun::map<SelectionInfo, InputImageP>(largeImages, [](const SelectionInfo& x) -> InputImageP { return x.image; }));
            }

            planarCorrelations = CorrespondenceCrossProduct(planarCorrelations);

            outSink.Push(GetAdjustedImages());
            outSink.Finish();
        }

        std::vector<InputImageP> GetAdjustedImages() const {
            std::vector<InputImageP> images;
            for(auto info : largeImages) {
                images.push_back(info.image);
            }
            return images;
        }

        const AlignmentGraph& GetAlignment() const {
            return alignment;
        }
        
        const ExposureCompensator& GetExposure() const {
            return exposure;
        }
    };
}

#endif

