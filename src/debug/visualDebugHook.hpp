#include "debugHook.hpp"
#include "../io/inputImage.hpp"
#include <irrlicht.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_VISUAL_DEBUG_HOOK
#define OPTONAUT_VISUAL_DEBUG_HOOK

namespace optonaut {

    struct DebugImage {
        cv::Mat image;
        cv::Mat position;
        cv::Mat orientation;
        float scale;
    };

    struct DebugFeature {
        double x, y, z; 
        int r, g, b;
    };

    class VisualDebugHook : public DebugHook {
        private: 
            irr::IrrlichtDevice *device; 
            irr::video::IVideoDriver* driver;
            irr::scene::ISceneManager* smgr;
            irr::gui::IGUIEnvironment* guienv;
            const irr::scene::IGeometryCreator* geoCreator;
            irr::scene::IMeshManipulator* meshManipulator;
            irr::scene::ICameraSceneNode* camera;

            //std::thread worker;
            //std::mutex m;
            //std::condition_variable cv;
            //bool isRunning;
            std::vector<DebugImage> asyncInput;
            std::vector<DebugFeature> asyncFeatures;

            void RegisterImageInternal(const DebugImage &image);
            void RegisterFeatureInternal(const DebugFeature &image);

            void Run();
        public:
            VisualDebugHook();
            void RegisterImage(const cv::Mat &image, const cv::Mat &position, const cv::Mat &orientation, float scale = 1);
            void RegisterImage(const cv::Mat &image, const cv::Mat &position, float scale = 1);
            void RegisterImageRotationModel(const cv::Mat &image, const cv::Mat &extrinsics, const cv::Mat &intrinsics, float scale = 1);

            void PlaceFeature(double x, double y, double z, int r = 0xFF, int g = 0x00, int b = 0x00);
            
            void WaitForExit();

            void Draw();
    };
}

#endif
