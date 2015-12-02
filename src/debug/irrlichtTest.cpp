#include <irrlicht.h>

using namespace irr;

using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

int main()
{
	IrrlichtDevice *device =
		createDevice( video::EDT_OPENGL, dimension2d<u32>(640, 480), 16,
			false, false, false, 0);

	if (!device)
		return 1;
	
    device->setWindowCaption(L"Hello World! - Irrlicht Engine Demo");

	IVideoDriver* driver = device->getVideoDriver();
	ISceneManager* smgr = device->getSceneManager();
	IGUIEnvironment* guienv = device->getGUIEnvironment();
    const IGeometryCreator* geoCreator = smgr->getGeometryCreator();
    IMeshManipulator* meshManipulator = smgr->getMeshManipulator();

    //ILogger* logger = device->getLogger(); 
    //logger->log(device->getFileSystem()->getWorkingDirectory().c_str());

	IAnimatedMesh* mesh = smgr->getMesh("debug/camera.3ds");
    meshManipulator->setVertexColors(mesh, SColor(0,255, 0x0, 0x0));
    IMesh* planeMesh = geoCreator->createPlaneMesh(core::dimension2d<f32>(1, 1), 
                                                   core::dimension2d<u32>(30, 30));
    IMesh* sphereMesh = geoCreator->createSphereMesh(1);

	if (!mesh || !planeMesh)
	{
		device->drop();
		return 1;
	}
	IAnimatedMeshSceneNode* node = smgr->addAnimatedMeshSceneNode(mesh);
	IMeshSceneNode* planeNode = smgr->addMeshSceneNode(planeMesh);
	IMeshSceneNode* sphereNode = smgr->addMeshSceneNode(sphereMesh);
    sphereNode->setPosition(vector3df(1, 0.3, 0.3));

	if (node || planeNode)
	{
		node->setMaterialFlag(EMF_LIGHTING, false);
        node->setMaterialFlag(video::EMF_WIREFRAME, true);
		
        planeNode->setMaterialFlag(EMF_LIGHTING, false);
        planeNode->setMaterialTexture(0, driver->getTexture("debug/texture.jpg"));
        planeNode->setMaterialFlag(video::EMF_BACK_FACE_CULLING, false);
        sphereNode->setMaterialFlag(EMF_LIGHTING, false);
        sphereNode->setMaterialFlag(video::EMF_WIREFRAME, true);
		//node->setMaterialTexture( 0, driver->getTexture("../../media/sydney.bmp") );
	}

	smgr->addCameraSceneNode(0, vector3df(0,10,-10), vector3df(0,0,0));
    //smgr->addCameraSceneNodeFPS();
    

    float rot = 0;

	while(device->run())
	{
		driver->beginScene(true, true, SColor(255,100,101,140));

		smgr->drawAll();
		guienv->drawAll();
        planeNode->setRotation(vector3df(0, 0, rot));
        planeNode->setScale(vector3df(rot / 1024 + 1, rot / 1024 + 1, 1));
        sphereNode->setRotation(vector3df(rot * 3, 0, 0));
        node->setRotation(vector3df(0, 0, rot * -4));

        rot += 0.1;

		driver->endScene();
	}

	device->drop();

	return 0;
}

