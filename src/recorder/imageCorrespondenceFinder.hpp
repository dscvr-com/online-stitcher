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

        const int downsample = 2;

        void ComputeMatch(const SelectionInfo &a, const SelectionInfo &b, 
                          int overlapArea) {
           int minSize = min(a.image->image.cols, b.image->image.rows) / 1.8;
           auto res = matcher.Match(a.image, b.image, minSize, minSize, false, 0.5);

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
                    aToB.dx = res.offset.x;
                    aToB.dy = res.offset.y;
                    aToB.overlap = res.correlationCoefficient * 2;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;
                    
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
            Sink<std::vector<InputImageP>> &outSink) : 
            outSink(outSink) { 
            warperFactory = new cv::SphericalWarper();
            warper = warperFactory->create(static_cast<float>(1600));
        }

        virtual void Push(SelectionInfo info) {

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
            
            for(auto cand : largeImages) {
                auto roiCand = GetOuterRectangle(*warper, cand.image);
                auto inCand = GetOuterRectangle(*warper, info.image);

                int overlapArea = (roiCand & inCand).area();

                if(overlapArea > inCand.area() / 6) {
                    ComputeMatch(info, cand, overlapArea);
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

            // Todo - not sure if apply is good here. 
            for(auto info : largeImages) {
                alignment.Apply(info.image);
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

