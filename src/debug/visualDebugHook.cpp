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

    static double baseData[] = 
        {1, 0, 0, 0, 
         0, -1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1};

    static const Mat base(4, 4, CV_64F, baseData);

    vector3df IrrVectorFromCVVector(const Mat &vec, vector<int> remap = vector<int>()) {
        AssertM(MatIs(vec, 3, 1, CV_64F), "Given Mat is a Vector");

        if(remap.size() == 0) {
            return vector3df(
                vec.at<double>(0),
                vec.at<double>(1),
                vec.at<double>(2));
        } else {
            return vector3df(
                vec.at<double>(remap[0]),
                vec.at<double>(remap[1]),
                vec.at<double>(remap[2]));
        }
    }

    ITexture* ImageToTexture(video::IImage* image, core::stringc name, IVideoDriver *driver) {
        video::ITexture* texture = driver->addTexture(name.c_str(),image);
        texture->grab();
        return texture;
    }

    void BRGToRGB(const cv::Mat &in, cv::Mat &out) {
        AssertEQM(in.cols, out.cols, "In/out have the same width");
        AssertEQM(in.rows, out.rows, "In/out have the same heigh");
        AssertEQM(in.type(), out.type(), "In/out have the same type");

        for(int i = 0; i < in.cols; i++) {
            for(int j = 0; j < in.rows; j++) {
                auto brg = in.at<cv::Vec3b>(j, i);
                std::swap(brg[0], brg[2]);
                out.at<cv::Vec3b>(j, i) = brg;
            }
        } 
    }
    
    void VisualDebugHook::RegisterFeatureInternal(const DebugFeature &f) {
        IMesh* sphereMesh = geoCreator->createSphereMesh(0.1, 5, 5);
        meshManipulator->setVertexColors(sphereMesh, SColor(0, f.r, f.g, f.b));
	    IMeshSceneNode* sphereNode = smgr->addMeshSceneNode(sphereMesh);
        sphereNode->setPosition(vector3df(f.x, f.y, f.z));
        sphereNode->setMaterialFlag(EMF_LIGHTING, false);
    }
    
    void VisualDebugHook::RegisterCameraInternal(const DebugCamera &cam) {

        IAnimatedMesh* cameraMesh = smgr->getMesh("src/debug/camera.3ds");
        std::string newMeshName = "cam-mesh-" + ToString(cam.camId);
        smgr->getMeshCache()->renameMesh(cameraMesh, newMeshName.c_str()); 
        cameraMesh = smgr->getMesh(newMeshName.c_str());

        meshManipulator->setVertexColors(cameraMesh, 
                SColor(0, std::min(0xFF, (int)cam.camId), 0x00, 0xFF));

        IMeshSceneNode* cameraNode = smgr->addMeshSceneNode(cameraMesh);

        cameraNode->setMaterialFlag(EMF_LIGHTING, false);
        cameraNode->setMaterialFlag(video::EMF_WIREFRAME, true);
        cameraNode->setMaterialFlag(video::EMF_BACK_FACE_CULLING, true);

        cameraNode->setPosition(vector3df(cam.x, cam.y, cam.z));
        cameraNode->setScale(vector3df(0.5, 0.5, 0.5));
        
        Mat rvec; ExtractRotationVector(cam.orientation.inv(), rvec);
        cameraNode->setRotation(IrrVectorFromCVVector(rvec * 180.0 / M_PI, 
                    {0, 1, 2}));
    }

    void VisualDebugHook::RegisterImageInternal(const DebugImage &in) {
        //std::unique_lock<std::mutex> lock(m);  

        float ratio = (float)in.image.rows / (float)in.image.cols;
        IMesh* planeMesh = geoCreator->createPlaneMesh(
                core::dimension2d<f32>(1 * in.scale, ratio * in.scale), 
                core::dimension2d<u32>(1, 1));

        matrix4 upTransform;
        upTransform.setRotationRadians(vector3df(M_PI / -2, 0, 0));

        meshManipulator->transform(planeMesh, upTransform); 
        
	    IMeshSceneNode* planeNode = smgr->addMeshSceneNode(planeMesh);
        //Swap green and red channel. 
        Mat rgbImage(in.image.rows, in.image.cols, CV_8UC3);
        BRGToRGB(in.image, rgbImage);
        IImage* irrImage = driver->createImageFromData(ECF_R8G8B8, dimension2d<u32>(in.image.cols, in.image.rows), rgbImage.data);
        ITexture* texture = ImageToTexture(irrImage, "Unnamed Texture", driver); 
        
        planeNode->setMaterialTexture(0, texture);

        planeNode->setMaterialFlag(EMF_LIGHTING, false);
        planeNode->setMaterialFlag(video::EMF_BACK_FACE_CULLING, false);

        planeNode->setPosition(IrrVectorFromCVVector(in.position));

        Mat rvec; ExtractRotationVector(in.orientation.inv(), rvec);
        planeNode->setRotation(IrrVectorFromCVVector(rvec * 180.0 / M_PI, {0, 1, 2}));
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

        camera = smgr->addCameraSceneNode(0, vector3df(0,0,0), vector3df(1,0,0));
        
        float camRotation = 0;

        for(auto img : asyncInput) {
            RegisterImageInternal(img);
        }

        for(auto feat : asyncFeatures) {
            RegisterFeatureInternal(feat);
        }
        
        for(auto cam : asyncCameras) {
            RegisterCameraInternal(cam);
        }

        while(device->run()) {
            driver->beginScene(true, true, SColor(255,0,0,0));
            camRotation += 0.01f;
            //camera->setTarget(vector3df(sin(camRotation), 0, cos(camRotation)));
            camera->setPosition(vector3df(sin(camRotation) * 50, 0, cos(camRotation) * 50));
            camera->setTarget(vector3df(0, 0, 0));
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
    
    void VisualDebugHook::RegisterCamera(const cv::Mat &orientation, 
                    double x, double y, double z, size_t camId) {
        asyncCameras.push_back({base * orientation.inv() * base.t(), 
                x, -y, z, camId});
    }
    
    void VisualDebugHook::RegisterImage(const cv::Mat &image, const cv::Mat &position, float scale) {
        Mat pconv(3, 1, CV_64F);
        position(Rect(0, 3, 1, 3)).copyTo(pconv);

        pconv.at<double>(1) *= -1;
        

        RegisterImage(image, 
               pconv,
               base * position(Rect(0, 0, 3, 3)) * base.t(), scale); 
    }
    
    void VisualDebugHook::RegisterImageRotationModel(const cv::Mat &image, const cv::Mat &extrinsics, const cv::Mat &intrinsics, float scale) {

        double dist[] = {0, 0, intrinsics.at<double>(0, 0), 1 };
        Mat pos = base * extrinsics * base.t() 
            * Mat(1, 4, CV_64F, dist).t();
       
        RegisterImage(image, 
                pos(Rect(0, 0, 1, 3)), 
                (base * extrinsics.t() * base.t())(Rect(0, 0, 3, 3)), 
                scale * intrinsics.at<double>(1, 2)); 
    }
            
    void VisualDebugHook::PlaceFeature(double x, double y, double z, int r, int g, int b) {
        asyncFeatures.push_back({x, y, z, r, g, b});
    }
    
    void VisualDebugHook::WaitForExit() {
        //worker.join();
    }
}
