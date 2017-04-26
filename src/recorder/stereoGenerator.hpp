#include "../common/sink.hpp"
#include "../common/ringProcessor.hpp"
#include "../recorder/imageSelector.hpp"
#include "../stereo/monoStitcher.hpp"
#include "../recorder/imageCorrespondenceFinder.hpp"

#ifndef OPTONAUT_STEREO_GENERATOR_HEADER
#define OPTONAUT_STEREO_GENERATOR_HEADER

namespace optonaut {
class StereoGenerator : public SelectionSink {
    private:
        ImageSink &leftOutputSink;
        ImageSink &rightOutputSink;
        const MonoStitcher stereoConverter;

        int lastRingId;

        RingProcessor<SelectionInfo> stereoRingBuffer;
        std::map<std::pair<size_t, size_t>, cv::Point2d> correctedOffsets;
        const RecorderGraph &graph;
        const ImageCorrespondenceFinder &correspondences;

        void ConvertToStereo(const SelectionInfo &a, const SelectionInfo &b) {
            StereoImage stereo;
            SelectionEdge dummy;

            bool hasEdge = graph.GetEdge(a.closestPoint, 
                b.closestPoint, dummy);
                
            AssertWM(hasEdge, "Pair is correctly ordered");
                
           // if(!hasEdge)
           //     return;
            
            bool hasOffset = false;
            Point2d offset;
            Log << "Looking for offset between " << a.image->id << " and " << b.image->id;
            auto offsets = correspondences.GetPlanarOffsets();
            auto it = offsets.find(std::make_pair(a.image->id, b.image->id));

            if(it != offsets.end()) {
                hasOffset = true;
                offset = it->second;
                Log << "Correcting offset: " << offset;
            }
           
            // TODO - this is slow! 
            AutoLoad alA(a.image), alB(b.image);
                
            stereoConverter.CreateStereo(a, b, stereo, offset);

            if(hasOffset) {
               // correctedOffsets.emplace(std::make_pair(a.image->id, b.image->id), offset);
               // Log << "Corrected offset: " << offset;
            }

            leftOutputSink.Push(stereo.A);
            rightOutputSink.Push(stereo.B);
       }
    public:
        StereoGenerator(
            ImageSink &leftOutputSink,
            ImageSink &rightOutputSink,
            const RecorderGraph &graph,
            const ImageCorrespondenceFinder &correspondences) : 
            leftOutputSink(leftOutputSink), rightOutputSink(rightOutputSink), 
            lastRingId(-1),
            stereoRingBuffer(1, std::bind(&StereoGenerator::ConvertToStereo, this, placeholders::_1, placeholders::_2), [](const SelectionInfo&) {}),
            graph(graph),
            correspondences(correspondences) {
        }
        virtual void Push(SelectionInfo image) {
            Log << "Received Image.";
            Log << "K: " << image.image->intrinsics;
            stereoRingBuffer.Push(image);
             
            if(lastRingId != -1 && lastRingId != (int)image.closestPoint.ringId) {
                stereoRingBuffer.Flush();
            }

            lastRingId = (int)image.closestPoint.ringId;
        }

        virtual void Finish() {
            stereoRingBuffer.Flush();
            leftOutputSink.Finish();
            rightOutputSink.Finish();
        }
};
}

#endif
