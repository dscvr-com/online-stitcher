#include "../math/stat.hpp"
#include "../recorder/alignmentGraph.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"

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

        const int downsample = 1;

        void ComputeMatch(const SelectionInfo &a, const SelectionInfo &b, 
                          int overlapArea) {
            int minSize = min(a.image->image.cols, b.image->image.rows) / 3;
            auto res = matcher.Match(a.image, b.image, minSize, minSize, false, 0.5);

            Log << "B adj extrinsics: " << b.image->adjustedExtrinsics;
            Log << "A adj extrinsics: " << a.image->adjustedExtrinsics;
            Log << "B orig extrinsics: " << b.image->originalExtrinsics;
            Log << "A orig extrinsics: " << a.image->originalExtrinsics;

            Log << "Computing match between " << a.image->id << " and " << b.image->id << ": " << res.valid;

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
                    aToB.overlap = res.correlationCoefficient * 2;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;

                    Log << "Offset " << a.image->id << 
                        " <> " << b.image->id << 
                        " : " << res.offset * std::pow(2, downsample);
                    
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
            }
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
            Sink<std::vector<InputImageP>> &outSink, const RecorderGraph &fullGraph) :
            outSink(outSink), graph(fullGraph) { 
            warperFactory = new cv::SphericalWarper();
            warper = warperFactory->create(static_cast<float>(1600));
        }

        virtual void Push(SelectionInfo info) {
            Log << "Received Image: " << info.image->id;
            Log << "K: " << info.image->intrinsics;

            // Downsample the iamge - create a minified copy. 
            auto miniCopy = std::make_shared<InputImage>(*(info.image));

            pyrDown(miniCopy->image.data, miniCopy->image.data);
            pyrDown(miniCopy->image.data, miniCopy->image.data);
                
            cv::Mat small; 
    
            pyrDown(miniCopy->image.data, small);

            for(int i = 1; i < downsample; i++) {
                pyrDown(small, small);
            }              

            miniCopy->image = Image(small); 
            
            SelectionInfo infoCopy = info;
            infoCopy.image = miniCopy;
            // Now match with all possible images.
            
            for(auto cand : miniImages) {
                auto roiCand = GetOuterRectangle(*warper, cand.image);
                auto inCand = GetOuterRectangle(*warper, infoCopy.image);

                int overlapArea = (roiCand & inCand).area();

                bool sameRing = cand.closestPoint.ringId == infoCopy.closestPoint.ringId;

                bool neighbors = sameRing && (
                        std::abs((int)cand.closestPoint.localId - (int)infoCopy.closestPoint.localId) == 1 ||
                        (size_t)std::abs((int)cand.closestPoint.localId - (int)infoCopy.closestPoint.localId) == graph.GetRings()[cand.closestPoint.ringId].size() - 1);


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
            double error = 0; 

            miniImages.clear();
            vector<InputImageP> images;
            for(auto info : largeImages) {
                images.push_back(info.image);
            }
            auto rings = graph.SplitIntoRings(images); 
            deque<double> adjustments; 

            int currentRing = rings.size() / 2;

            // What's missing: Matching between rings. Proper adjustment of whole graph. 
            AssertEQM(rings.size(), (size_t)1, "Only single ring supported at the moment");

            // First, close all rings.
            // Do so in recording order
            while(currentRing >= 0 && currentRing < (int)rings.size()) {
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

                        for(auto &fwdEdge : alignment.GetEdges()[ring[i]->id]) {
                            fwdEdge.value.dphi -= ydiff;
                            auto _bwEdges = alignment.GetEdges(fwdEdge.from, fwdEdge.to);
                            bwEdges.insert(bwEdges.begin(), _bwEdges.begin(), _bwEdges.end());

                        }

                        std::sort(bwEdges.begin(), bwEdges.end() );
                        bwEdges.erase(unique(bwEdges.begin(), bwEdges.end()), bwEdges.end());
                        for(auto bwEdge : bwEdges) {
                            bwEdge->value.dphi += ydiff;
                        }
                    }

                    //4) TODO: correct graph for other rings. Alignment is probably sane, but
                    //it's way more efficient to "pre-turn" the rings. 
                    //DO NOT correct focal len
                }
                currentRing = graph.GetNextRing(currentRing);
            }

            // Second, perform global optimization. 
            alignment.FindAlignment(error); 
            exposure.FindGains();

            // Third, apply corrections.  
            for(size_t i = 0; i < rings.size(); i++) {
                auto ring = rings[i];
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

                double focalLenAdjustment = (1 - dist / (M_PI * 2));
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

            // Todo - not sure if apply is good here. 
            for(auto info : largeImages) {
                // TODO - Apply has no effect!
                alignment.Apply(info.image);
                info.image->adjustedExtrinsics.copyTo(info.image->originalExtrinsics);
                // Exposure disabled for now. 
                // exposure.Apply(info.image->image.data, info.image->id);
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

