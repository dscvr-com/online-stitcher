#ifndef OPTONAUT_STEREO_GENERATOR_HEADER
#define OPTONAUT_STEREO_GENERATOR_HEADER

namespace optonaut {
class StereoGenerator : public SelectionSink {
    private:
        ImageSink &leftOutputSink;
        ImageSink &rightOutputSink;
        const MonoStitcher stereoConverter;

        int lastRingId;
        std::function<void(const SelectionInfo&, const SelectionInfo &b)> 
            ConvertToStereo; 
        RingProcessor<SelectionInfo> stereoRingBuffer;
    public:
        StereoGenerator(
            ImageSink &leftOutputSink,
            ImageSink &rightOutputSink,
            const RecorderGraph &graph) : 
            leftOutputSink(leftOutputSink), rightOutputSink(rightOutputSink), 
            lastRingId(-1),
            ConvertToStereo([&](const SelectionInfo &a, const SelectionInfo &b) {
                StereoImage stereo;
                SelectionEdge dummy;
                    
                bool hasEdge = graph.GetEdge(a.closestPoint, 
                    b.closestPoint, dummy);
                    
                AssertWM(hasEdge, "Pair is correctly ordered");
                    
                if(!hasEdge)
                    return;
               
                // TODO - this is slow! 
                AutoLoad alA(a.image), alB(b.image);
                    
                stereoConverter.CreateStereo(a, b, stereo);

                leftOutputSink.Push(stereo.A);
                rightOutputSink.Push(stereo.B);
            }),
            stereoRingBuffer(1, ConvertToStereo, [](const SelectionInfo&) {}) {
           
        }
        virtual void Push(SelectionInfo image) {
            Log << "Received Image.";
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

