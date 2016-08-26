#ifndef OPTONAUT_COORDINATE_CONVERTER_HEADER
#define OPTONAUT_COORDINATE_CONVERTER_HEADER

namespace optonaut {
class CoordinateConverter : public ImageSink {
    private:
        const Mat &base;
        const Mat &zero;
        const Mat baseInv;
        
        ImageSink &outputSink;
    public:
        CoordinateConverter(
            const Mat &base, const Mat &zeroWithoutBase, 
            ImageSink &outputSink) :
            base(base), zero(zeroWithoutBase), baseInv(base.inv()), 
            outputSink(outputSink) {
        }

        virtual void Push(InputImageP image) {
            Assert(image != NULL);

            // Explicitiely copy extrinsics to newly allocated mat 
            // to avoid mem leaks
            Mat extrinsics;
            ConvertToStitcher(image->originalExtrinsics, extrinsics);

            image->originalExtrinsics = Mat(4, 4, CV_64F);
            image->adjustedExtrinsics = Mat(4, 4, CV_64F);

            extrinsics.copyTo(image->originalExtrinsics);
            extrinsics.copyTo(image->adjustedExtrinsics);

            outputSink.Push(image); 
        }
        
        Mat ConvertFromStitcher(const Mat &in) const {
            return (zero.inv() * baseInv * in * base).inv();
        }
        
        void ConvertToStitcher(const Mat &in, Mat &out) const {
            out = (base * zero * in.inv() * baseInv);
        }

        virtual void Finish() {
            outputSink.Finish();
        }
    };
}

#endif

