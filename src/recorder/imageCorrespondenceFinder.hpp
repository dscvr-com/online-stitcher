#include "../math/stat.hpp"

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

        const RecorderGraph &graph;

        const int downsample = 2;

        void ComputeMatch(const SelectionInfo &a, const SelectionInfo &b, 
                          int overlapArea) {
           int minSize = min(a.image->image.cols, b.image->image.rows) / 1.8;
           auto res = matcher.Match(a.image, b.image, minSize, minSize, false, 0.5);

           Log << "Computing match between " << a.image->id << " and " << b.image->id;

           if(res.valid) {
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
    public:
        ImageCorrespondenceFinder(
            Sink<std::vector<InputImageP>> &outSink, const RecorderGraph &fullGraph) :
            outSink(outSink), graph(fullGraph) { 
            warperFactory = new cv::SphericalWarper();
            warper = warperFactory->create(static_cast<float>(1600));
        }

        virtual void Push(SelectionInfo info) {
            Log << "Received Image.";

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

                if(overlapArea > inCand.area() / 2) {
                    ComputeMatch(infoCopy, cand, overlapArea);
                }
            } 
            
            miniImages.push_back(infoCopy);
            largeImages.push_back(info);
        }

        virtual void Finish() {
            double error = 0; 
            alignment.FindAlignment(error); 
            exposure.FindGains();

            miniImages.clear();
            vector<InputImageP> images;
            for(auto info : largeImages) {
                images.push_back(info.image);
            }
            auto rings = graph.SplitIntoRings(images); 

            deque<double> adjustments; 

            for(size_t i = 0; i < rings.size(); i++) {
                auto ring = rings[i];
                double dist; 

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
               info.image->intrinsics.at<double>(0, 0) *= focalLenAdjustment; 
               info.image->intrinsics.at<double>(1, 1) *= focalLenAdjustment; 
            }

            // Todo - not sure if apply is good here. 
            for(auto info : largeImages) {
                alignment.Apply(info.image);
                info.image->adjustedExtrinsics.copyTo(info.image->originalExtrinsics);
                exposure.Apply(info.image->image.data, info.image->id);
            }

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

