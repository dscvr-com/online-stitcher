#include <opencv2/core.hpp>

#ifndef OPTONAUT_FLOW_BLENDER_HEADER
#define OPTONAUT_FLOW_BLENDER_HEADER

namespace optonaut {
    
    /*
     * Flow blender. We don't inherit from OpenCV's blender, since we want to save
     * memory by using U8C3 instead of U16C3.
     * This blending technique can handle small parallax well.
     * Restriction: Only two images may overlap at any point for a good blend.
     */
    class FlowBlender {
    public:
        FlowBlender(float sharpness = 0.005f);
        
        float GetSharpness() const { return sharpness; }
        void SetSharpness(float val) { sharpness = val; }
        
        void Prepare(const cv::Rect &dstRoi);
        void Feed(const cv::Mat &img, const cv::Mat &flow, const cv::Point &tl);
        void CalculateFlow(
                           const Mat &a, const Mat &b,
                           const Point &aTl, const Point &bTl,
                           Mat &flow,
                           Point &offset = dummyFlow, const bool reCalcOffset = true) const;
        const cv::Mat& GetResult() const { return dest; }
        const cv::Mat& GetResultMask() const { return destMask; }
        
    private:
        static Point dummyFlow;
        float sharpness;
        cv::Mat dest;
        cv::Mat destMask;
        cv::Rect destRoi;
        std::vector<cv::Rect> existingCores;
    };
    
    inline FlowBlender::FlowBlender(float sharpness) { SetSharpness(sharpness); }
}

#endif