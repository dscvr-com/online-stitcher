#ifndef OPTONAUT_IMAGE_LOADER_HEADER
#define OPTONAUT_IMAGE_LOADER_HEADER

namespace optonaut {
class ImageLoader : public ImageSink {
    private:
        ImageSink &outputSink;
    public:
        ImageLoader(ImageSink &outputSink):
            outputSink(outputSink) {
        }
        virtual void Push(InputImageP image) {
            Assert(image != NULL);
            Log << "Received Image.";

            if(!image->IsLoaded()) {
                image->LoadFromDataRef();
            }
            outputSink.Push(image);
        }

        virtual void Finish() {
            Log << "Finish";
            outputSink.Finish();
        }
    };
}

#endif

