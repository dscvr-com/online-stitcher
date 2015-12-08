#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_DEBUG_HOOK
#define OPTONAUT_DEBUG_HOOK

namespace optonaut {

    class DebugHook {
    public: 
        /**
         * Adds an image to debug output.
         *
         * @param image       The image to draw. 
         * @param position    The position as vector3, carthesian coords. 
         * @param orientation The orientation as rotation matrix3
         * @param scale       The scaling of the image. 
         */
        virtual void RegisterImage(const cv::Mat &image, const cv::Mat &position, const cv::Mat &orientation, float scale = 1) = 0;
        /**
         * Adds an image to debug output.
         *
         * @param image       The image to draw. 
         * @param position    The position as affine matrix4, containing rotation and translation. 
         * @param scale       The scaling of the image. 
         */
        virtual void RegisterImage(const cv::Mat &image, const cv::Mat &position, float scale = 1) = 0;
        
        /**
         * Adds an image to debug output.
         *
         * @param image       The image to draw. 
         * @param extrinsics  Extrinsic camera params, matrix4 consisting of rotation and translation.
         * @param intrinsics  Intrinsic camera params. 
         */
        virtual void RegisterImageRotationModel(const cv::Mat &image, const cv::Mat &extrinsics, const cv::Mat &intrinsics, float scale = 1) = 0;

        virtual void PlaceFeature(double x, double y, double z, int r = 0xFF, int g = 0x00, int b = 0x00) = 0;

        static DebugHook* Instance;

    };
}

#endif
