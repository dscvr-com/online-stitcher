#include <functional>
#include <deque>
#include "sink.hpp"
#include "../recorder/storageImageSink.hpp"

using namespace std;

#ifndef OPTONAUT_IMAGE_QUEUE_PROCESSOR_HEADER
#define OPTONAUT_IMAGE_QUEUE_PROCESSOR_HEADER

namespace optonaut {
    template <typename InType>
	class ImageQueueProcessor : public Sink<InType> {
	private:
        StorageImageSink &out;
    public:
        vector<InputImageP> postImages;
        ImageQueueProcessor(StorageImageSink &out ):
                out(out)
                {
        }
 
        bool HasResults() {
            return postImages.size() > 0;
        }

        virtual void Push(InType in) {
            Log << "ImageQueueProcessor.push()";
            out.Push(in);
            auto copy = std::make_shared<InputImage>(*in.image);
            copy->image = Image();
            postImages.push_back(copy);

        }

   

        // Alias for flush.
        virtual void Finish() {
            Log << "ImageQueueProcessor.Finish()";
            //out.Finish();
        }   
    };
}
#endif
