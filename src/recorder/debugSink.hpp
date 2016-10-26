#include "../io/io.hpp"

#ifndef OPTONAUT_DEBUG_SINK_HEADER
#define OPTONAUT_DEBUG_SINK_HEADER

namespace optonaut {
class DebugSink : public Sink<SelectionInfo> {
    private:
        const std::string debugPath;
        const bool forwardToOutput;
        Sink<SelectionInfo> &outputSink;
        
        void SaveOutput(const InputImageP image) {
            InputImageToFile(image, debugPath + ToString(image->id) + ".jpg");
        }
        
        void SaveOutput(const SelectionInfo image) {
            SaveOutput(image.image);
        }

    public:
        DebugSink(
            const std::string debugPath, 
            const bool forwardToOutput,
            Sink<SelectionInfo> &outputSink) :
            debugPath(debugPath), forwardToOutput(forwardToOutput),
            outputSink(outputSink) {
        }

        virtual void Push(SelectionInfo image) {
            
            if(debugPath.size() > 0) {
                SaveOutput(image);
            }

            if(forwardToOutput) {
                outputSink.Push(image);
            }
        }

        virtual void Finish() {

            if(forwardToOutput) {
                outputSink.Finish();
            } else {
                Log << "We're in debug mode. Crashing now.";
                std::abort();
            }
        }
    };
}

#endif

