#include <irrlicht.h>
#include <thread>

using namespace irr;

using namespace core;
using namespace scene;
using namespace video;

void runExample()
{
	IrrlichtDevice *device =
		createDevice( video::EDT_OPENGL, dimension2d<u32>(640, 480), 16,
			false, false, false, 0);

	IVideoDriver* driver = device->getVideoDriver();
	ISceneManager* smgr = device->getSceneManager();
    const IGeometryCreator* geoCreator = smgr->getGeometryCreator();

    IMesh* sphereMesh = geoCreator->createSphereMesh(1);
	IMeshSceneNode* sphereNode = smgr->addMeshSceneNode(sphereMesh);

    sphereNode->setMaterialFlag(EMF_LIGHTING, false);
    sphereNode->setMaterialFlag(video::EMF_WIREFRAME, true);

	smgr->addCameraSceneNode(0, vector3df(0,10,0), vector3df(0,0,0));
    
    float rot = 0;

	while(device->run())
	{
		driver->beginScene(true, true, SColor(255,100,101,140));

		smgr->drawAll();
        
        sphereNode->setRotation(vector3df(rot * 3, 0, 0));

        rot += 0.1;
		driver->endScene();
	}

	device->drop();
}

int main() {
    //Works
    //runExample();

    //Does not work
    auto worker = std::thread(runExample);
    worker.join();

    return 0;
}
