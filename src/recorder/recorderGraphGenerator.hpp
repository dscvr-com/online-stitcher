#include <opencv2/opencv.hpp>
#include <math.h>

#include "../io/inputImage.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"
#include "../common/ringProcessor.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_IMAGE_SELECTOR_HEADER
#define OPTONAUT_IMAGE_SELECTOR_HEADER

namespace optonaut {

class RecorderGraphGenerator {

private:
	//adj[n] contains m if m is right of n
	//Horizontal and Vertical overlap in percent. 
	const double hOverlap = 0.9;
	const double vOverlap = 0.25;
    
    Mat intrinsics;

public:
    
    RecorderGraph Generate(const Mat &intrinsics, const int mode = RecorderGraph::ModeAll) {
        
        RecorderGraph res;

		this->intrinsics = intrinsics;
        double maxHFov = GetHorizontalFov(intrinsics);
        double maxVFov = GetVerticalFov(intrinsics); 
		double hFov = maxHFov * (1.0 - hOverlap);
		double vFov = maxVFov * (1.0 - vOverlap);

        cout << "H-FOV: " << (maxHFov * 180 / M_PI) << endl;
        cout << "V-FOV: " << (maxVFov * 180 / M_PI) << endl;
        cout << "Ratio: " << (sin(maxVFov) / sin(maxHFov)) << endl;

        uint32_t vCount = ceil(M_PI / vFov);
        double vStart = 0;
        uint32_t id = 0;
        uint32_t hCenterCount = ceil(2 * M_PI / hFov);
        
        //TODO: Wrong assumptions.
        //This actually builds the recorder graph from the bottom. 

        if(mode == RecorderGraph::ModeTruncated || mode == RecorderGraph::ModeTinyDebug) {
           //Configuration for ModeTruncated
           //Optimize for 3 rings.
           
           vCount = vCount - 2; //Always skip out two rings.
           //vFov stays the same.
           vStart = (M_PI - (vFov * 3)) / 2;
            
        }
        else if(mode == RecorderGraph::ModeNoBot) {
            //Configuration for ModeNoBot
            vCount = vCount - 1;
            vStart = (M_PI - (vFov * 3)) / 2;
            
        } else if(mode == RecorderGraph::ModeCenter) {
            assert(true); //Not implemented and probably not needed any more.
        } else {
            //Configuration for ModeAll
            vStart = maxVFov * vOverlap;
            
            vFov = (M_PI - 2 * vStart) / vCount;
        }

        if(vCount % 2 == 0 && mode == RecorderGraph::ModeCenter) {
            cout << "Center mode not possible with even number of rings." << endl;
            assert(false);
        }

		for(uint32_t i = 0; i < vCount; i++) {

            //Vertical center, bottom and top of ring
			double vCenter = i * vFov + vFov / 2.0 - M_PI / 2.0 + vStart;

			uint32_t hCount = hCenterCount * cos(vCenter);
			hFov = M_PI * 2 / hCount;

            double hLeft = 0;
            SelectionEdge edge;
            
            if(mode ==  RecorderGraph::ModeTinyDebug) {
                hCount = 6;
            }

			res.targets.push_back(vector<SelectionPoint>(hCount));

            auto createEdge = [&res] (SelectionPoint &a, SelectionPoint &b) {
                SelectionEdge edge;
                edge.from = a.globalId;
                edge.to = b.globalId;
                edge.recorded = false;

                res.adj[edge.from].push_back(edge);
            };

            auto finish = [&res] (SelectionPoint &a) {
                res.targets[a.ringId][a.localId] = a;
            };

            RingProcessor<SelectionPoint> hqueue(1, createEdge, finish);
                    
            for(uint32_t j = 0; j < hCount; j++) {
                res.adj.push_back(vector<SelectionEdge>());
                
                hLeft = j * hFov;
                
                SelectionPoint p;
                p.globalId = id;
                p.hPos = hLeft;
                p.vPos = vCenter;
                p.ringId = i;
                p.localId = j;
                p.vFov = vFov;
                p.hFov = hFov;
                
                GeoToRot(hLeft, vCenter, p.extrinsics);
                
                hqueue.Push(p);
                
                id++;
            }

            hqueue.Flush();
            
		}

        return res;
	}
};
}

#endif
