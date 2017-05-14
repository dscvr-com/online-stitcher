#include "../io/io.hpp"

#ifndef OPTONAUT_DEBUG_SINK_HEADER
#define OPTONAUT_DEBUG_SINK_HEADER

namespace optonaut {
class DebugSink : public Sink<InputImageP> {
    private:
        const std::string debugPath;
        const bool forwardToOutput;
        Sink<InputImageP> &outputSink;
        
        void SaveOutput(const InputImageP image) {
            InputImageToFile(image, debugPath + ToString(image->id) + ".jpg");
        }

    public:
        DebugSink(
            const std::string debugPath, 
            const bool forwardToOutput,
            Sink<InputImageP> &outputSink) :
            debugPath(debugPath), forwardToOutput(forwardToOutput),
            outputSink(outputSink) {
        }

        virtual void Push(InputImageP image) {
            
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

