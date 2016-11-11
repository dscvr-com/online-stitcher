#include "../math/stat.hpp"
#include "../recorder/alignmentGraph.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"
#include "imageSink.hpp"

#ifndef OPTONAUT_IMAGE_FOR_POST_PROCESS_HEADER
#define OPTONAUT_IMAGE_FOR_POST_PROCESS_HEADER

namespace optonaut {
class ImageForPostProcess : public SelectionSink {
    private: 
        std::deque<SelectionInfo> largeImages;
        std::vector<SelectionInfo> miniImages;

        cv::Ptr<cv::WarperCreator> warperFactory;
        cv::Ptr<cv::detail::RotationWarper> warper;

        AlignmentGraph alignment;
        ExposureCompensator exposure;
        PairwiseCorrelator matcher;

        ImageSink &outSink;

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
        ImageForPostProcess(
            ImageSink &outSink,
            const RecorderGraph &fullGraph) :
            outSink(outSink), graph(fullGraph) { 
            warperFactory = new cv::SphericalWarper();
            warper = warperFactory->create(static_cast<float>(1600));
        }

        virtual void Push(SelectionInfo info) {
            Log << "Received Image: " << info.image->id;
            Log << "do nothing: ";
            outSink.Push(info);
            return ;

        }

        const std::map<std::pair<size_t, size_t>, cv::Point2d>& GetPlanarOffsets() const {
            return planarCorrelations;
        }

        virtual void Finish() {

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

