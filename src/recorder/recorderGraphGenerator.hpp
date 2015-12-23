#include <opencv2/opencv.hpp>
#include <math.h>

#include "../io/inputImage.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/biMap.hpp"
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
	static constexpr double hOverlap_ = 0.9;
	static constexpr double vOverlap_ = 0.25;

    static const bool debug = false;
    
    static void CreateEdge(RecorderGraph &res, 
            const SelectionPoint &a, const SelectionPoint &b) {
        SelectionEdge edge;
        edge.from = a.globalId;
        edge.to = b.globalId;
        edge.recorded = false;

        res.AddEdge(edge);
    };

    static void AddNode(RecorderGraph &res, const SelectionPoint &a) {
        res.AddNewNode(a); 
    };
public:

    static RecorderGraph Sparse(const RecorderGraph &in, BiMap<uint32_t, uint32_t> &denseToSparse, int skip) {

        RecorderGraph sparse(in.ringCount, in.intrinsics);

        size_t globalId = 0;
        size_t localId = 0;

        const auto &rings = in.GetRings();
        for(size_t i = 0; i < rings.size(); i++) {
            sparse.AddNewRing(rings[i].size() / skip);
            localId = 0;
            
            RingProcessor<SelectionPoint> hqueue(1, 
                    bind(CreateEdge, sparse, placeholders::_1, placeholders::_2),
                    bind(AddNode, sparse, placeholders::_1));

            for(size_t j = 0; j < rings[i].size(); j += skip) {
                SelectionPoint copy = rings[i][j];
                
                denseToSparse.Insert(copy.globalId, globalId);

                copy.localId = localId;
                copy.globalId = globalId;

                hqueue.Push(copy);

                localId++;
                globalId++;
            }
        }

        return sparse;
    }
    
    static RecorderGraph Generate(const Mat &intrinsics, const int mode = RecorderGraph::ModeAll, const float density = RecorderGraph::DensityNormal, const int lastRingOverdrive = 0) {
        AssertWGEM(density, 1.f, 
                "Reducing recorder graph density below one is potentially unsafe.");
        
        double hOverlap = 1.0 - (1.0 - hOverlap_) / density;
        double vOverlap = vOverlap_;

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
           
           vCount = 3; 
           //vFov stays the same.
           vStart = (M_PI - (vFov * 3)) / 2;
            
        }
        else if(mode == RecorderGraph::ModeNoBot) {
            //Configuration for ModeNoBot
            //vCount = vCount - 1;
            vCount = vCount - 1;
            vStart = M_PI - (vFov * vCount);
            
        } else if(mode == RecorderGraph::ModeCenter) {
            vCount = 1;
            vStart = M_PI / 2 - (vFov / 2);
        } else {
            //Configuration for ModeAll
            vStart = maxVFov * vOverlap;
            vFov = (M_PI - 2 * vStart) / vCount;
        }

        if(vCount % 2 == 0 && mode == RecorderGraph::ModeCenter) {
            cout << "Center mode not possible with even number of rings." << endl;
            assert(false);
        }
        
        RecorderGraph res(vCount, intrinsics);
        
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
            int ringOverdrive = 0;
            
            if(i == vCount -1) {
                ringOverdrive = lastRingOverdrive;
            }
            
            hCount = hCount + ringOverdrive;
            AssertEQ(i, res.AddNewRing(hCount));

            RingProcessor<SelectionPoint> hqueue(1, 
                    bind(CreateEdge, std::ref(res), placeholders::_1, placeholders::_2),
                    bind(AddNode, std::ref(res), placeholders::_1));
 
            for(uint32_t j = 0; j < hCount; j++) {
                if(debug) {
                    cout << "Recorder Graph Pushing " << hLeft << ", " << vCenter << endl;
                }

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
