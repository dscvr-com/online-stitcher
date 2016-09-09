#ifndef OPTONAUT_IMAGE_RESELECTOR_HEADER
#define OPTONAUT_IMAGE_RESELECTOR_HEADER

namespace optonaut {
class ImageReselector : public Sink<std::vector<InputImageP>> {
    private:
        SelectionSink &outputSink;
        const RecorderGraph &graph;
    public:
        ImageReselector(
            SelectionSink &outputSink,
            const RecorderGraph &graph) : 
            outputSink(outputSink), graph(graph) { }

        virtual void Push(std::vector<InputImageP> images) {
            Log << "Received Image.";
            BiMap<size_t, uint32_t> finalImagesToTargets;

            vector<InputImageP> bestAlignment = 
                graph.SelectBestMatches(images, finalImagesToTargets, false);
            
            sort(bestAlignment.begin(), bestAlignment.end(), 
            [&] (const InputImageP &a, const InputImageP &b) {
                uint32_t aId, bId;
                Assert(finalImagesToTargets.GetValue(a->id, aId));
                Assert(finalImagesToTargets.GetValue(b->id, bId));

                return aId < bId;
            });

            Log << " Use " << bestAlignment.size() << "/" << graph.Size() << " from " << images.size() << " input images.";
            
            for(auto img : bestAlignment) {
            	SelectionInfo info;
            	
                //Reassign points
            	uint32_t pointId = 0;
            	Assert(finalImagesToTargets.GetValue(img->id, pointId));
            	Assert(graph.GetPointById(pointId, info.closestPoint));

                double maxVFov = GetVerticalFov(img->intrinsics);
                info.closestPoint.vFov = maxVFov;

            	info.isValid = true;
            	info.image = img;

                outputSink.Push(info);
            }
        }

        virtual void Finish() {
            outputSink.Finish();
        }
    };
}

#endif

