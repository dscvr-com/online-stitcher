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

        double hBufferRatio;
        double vBufferRatio;
        
        void ConvertToStereo(const SelectionInfo &a, const SelectionInfo &b) {
            StereoImage stereo;
            SelectionEdge dummy;

            bool hasEdge = graph.GetEdge(a.closestPoint, 
                b.closestPoint, dummy);
                
            AssertWM(hasEdge, "Pair is correctly ordered");
                
            // TODO - this is slow! 
            AutoLoad alA(a.image), alB(b.image);
                
            stereoConverter.CreateStereo(a, b, stereo, hBufferRatio, vBufferRatio);

            leftOutputSink.Push(stereo.A);
            rightOutputSink.Push(stereo.B);
       }
    public:
        StereoGenerator(
            ImageSink &leftOutputSink,
            ImageSink &rightOutputSink,
            const RecorderGraph &graph, 
            double hBufferRatio = 1, 
            double vBufferRatio = -0.05) :
            leftOutputSink(leftOutputSink), rightOutputSink(rightOutputSink), 
            lastRingId(-1),
            stereoRingBuffer(1, std::bind(&StereoGenerator::ConvertToStereo, this, placeholders::_1, placeholders::_2), [](const SelectionInfo&) {}),
            graph(graph),
            hBufferRatio(hBufferRatio),
            vBufferRatio(vBufferRatio) {
        }
        virtual void Push(SelectionInfo image) {
            Log << "Received Image.";
            Log << "K: " << image.image->intrinsics;
             
            if(lastRingId != -1 && lastRingId != (int)image.closestPoint.ringId) {
                stereoRingBuffer.Flush();
            }
            
            stereoRingBuffer.Push(image);

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
