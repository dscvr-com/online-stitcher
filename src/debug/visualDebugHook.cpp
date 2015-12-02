#include "visualDebugHook.hpp"
#include "../common/assert.hpp"
#include "../math/support.hpp"

using namespace irr;
using namespace cv;
using namespace std;

using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

namespace optonaut {

    vector3df IrrVectorFromCVVector(const Mat &vec) {
        AssertM(MatIs(vec, 3, 1, CV_64F), "Given Mat is a Vector");

        return vector3df(
            vec.at<double>(0),
            vec.at<double>(1),
            vec.at<double>(2));
    }

    ITexture* ImageToTexture(video::IImage* image, core::stringc name, IVideoDriver *driver) {
        video::ITexture* texture = driver->addTexture(name.c_str(),image);
        texture->grab();
        return texture;
    }
            
    void VisualDebugHook::RegisterImageInternal(const DebugImage &in) {
        //std::unique_lock<std::mutex> lock(m);  

        float ratio = (float)in.image.rows / (float)in.image.cols;
        IMesh* planeMesh = geoCreator->createPlaneMesh(
                core::dimension2d<f32>(1 * in.scale, ratio * in.scale), 
                core::dimension2d<u32>(1, 1));
        
	    IMeshSceneNode* planeNode = smgr->addMeshSceneNode(planeMesh);
        IImage* irrImage = driver->createImageFromData(ECF_R8G8B8, dimension2d<u32>(in.image.cols, in.image.rows), in.image.data);
        ITexture* texture = ImageToTexture(irrImage, "Unnamed Texture", driver); 
        
        planeNode->setMaterialTexture(0, texture);

        planeNode->setMaterialFlag(EMF_LIGHTING, false);
        planeNode->setMaterialFlag(video::EMF_BACK_FACE_CULLING, false);

        planeNode->setPosition(IrrVectorFromCVVector(in.position));

        Mat rvec; ExtractRotationVector(in.orientation, rvec);
        planeNode->setRotation(IrrVectorFromCVVector(rvec));
    }

    void VisualDebugHook::Run() {
        device =
            createDevice(video::EDT_OPENGL, core::dimension2d<u32>(640, 480), 16,
                false, false, false, 0);

        driver = device->getVideoDriver();
        smgr = device->getSceneManager();
        guienv = device->getGUIEnvironment();
        geoCreator = smgr->getGeometryCreator();
        meshManipulator = smgr->getMeshManipulator();

        smgr->addCameraSceneNode(0, vector3df(0,10,0), vector3df(0,0,0));

        for(auto img : asyncInput) {
            RegisterImageInternal(img);
        }

        while(device->run()) {
            driver->beginScene(true, true, SColor(255,100,101,140));

            smgr->drawAll();
            guienv->drawAll();

            driver->endScene();
        } 

        device->drop();
    }

    VisualDebugHook::VisualDebugHook() {
    }
           
    void VisualDebugHook::Draw() {
        Run();
        /*
        worker = thread(&VisualDebugHook::Run, this);

        {
            std::unique_lock<std::mutex> lock(m);  
            cv.wait(lock);
        }*/ 
    }

    void VisualDebugHook::RegisterImage(const cv::Mat &image, const cv::Mat &position, const cv::Mat &orientation, float scale) {
        asyncInput.push_back({image, position, orientation, scale});
    }
    
    void VisualDebugHook::RegisterImage(const cv::Mat &image, const cv::Mat &position, float scale) {
       RegisterImage(image, position(Rect(0, 3, 1, 3)), position(Rect(0, 0, 3, 3)), scale); 
    }
    
    void VisualDebugHook::RegisterImageRotationModel(const cv::Mat &image, const cv::Mat &extrinsics, const cv::Mat &intrinsics, float scale) {
        double dist[] = {intrinsics.at<double>(0, 0), 0, 0, 1 };
        Mat pos = extrinsics * Mat(1, 4, CV_64F, dist).t();
        
        RegisterImage(image, pos(Rect(0, 0, 1, 3)), extrinsics(Rect(0, 0, 3, 3)), scale); 
    }
    
    void VisualDebugHook::WaitForExit() {
        //worker.join();
    };
}
