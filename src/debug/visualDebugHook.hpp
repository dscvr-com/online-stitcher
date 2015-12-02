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

    class VisualDebugHook : DebugHook {
        private: 
            irr::IrrlichtDevice *device; 
            irr::video::IVideoDriver* driver;
            irr::scene::ISceneManager* smgr;
            irr::gui::IGUIEnvironment* guienv;
            const irr::scene::IGeometryCreator* geoCreator;
            irr::scene::IMeshManipulator* meshManipulator;

            //std::thread worker;
            //std::mutex m;
            //std::condition_variable cv;
            //bool isRunning;
            std::vector<DebugImage> asyncInput;

            void RegisterImageInternal(const DebugImage &image);

            void Run();
        public:
            VisualDebugHook();
            void RegisterImage(const cv::Mat &image, const cv::Mat &position, const cv::Mat &orientation, float scale = 1);
            void RegisterImage(const cv::Mat &image, const cv::Mat &position, float scale = 1);
            void RegisterImageRotationModel(const cv::Mat &image, const cv::Mat &extrinsics, const cv::Mat &intrinsics, float scale = 1);
            
            void WaitForExit();

            void Draw();
    };
}

#endif
