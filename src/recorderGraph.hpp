
#include <opencv2/opencv.hpp>
#include <math.h>

#include "image.hpp"
#include "support.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_RECORDER_GRAPH_HEADER
#define OPTONAUT_RECORDER_GRAPH_HEADER

namespace optonaut {
    
    struct SelectionPoint {
        uint32_t globalId;
        uint32_t localId;
        uint32_t ringId;
        bool enabled;
        Mat extrinsics;
        
        SelectionPoint() : globalId(0),
            localId(0), ringId(0), enabled(false) {
        }
    };
    
    struct SelectionEdge {
        uint32_t from;
        uint32_t to;
        
        Mat roiCorners[4];
        Mat roiCenter;
        
        SelectionEdge() : from(0), to(0) {
            
        }
    };
    
    //Todo - Declare Friend with recorder Graph Generator, then make props private
    class RecorderGraph {
    public:
        
        static const int ModeAll = 0;
        static const int ModeCenter = 1;
        static const int ModeTruncated = 2;
        
        vector<vector<SelectionEdge>> adj;
        vector<vector<SelectionPoint>> targets;
        Mat intrinsics;
        
        const vector<vector<SelectionPoint>> &GetRings() const {
            return targets;
        }
        
        void RemoveEdge(const SelectionPoint& left, const SelectionPoint& right) {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    adj[left.globalId].erase(it);
                    break;
                }
            }
        }
        
        bool HasEdge(const SelectionPoint& left, const SelectionPoint& right, SelectionEdge &outEdge) const {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    outEdge = *it;
                    return true;
                }
            }
            return false;
        }
        
        bool GetNext(const SelectionPoint &current, SelectionPoint &next) const {
            auto it = adj[current.globalId].begin();
            
            if(adj[current.globalId].end() != it)
            {
                return GetPointById(it->to, next);
            }
            
            return false;
        }
        
        bool GetPointById(uint32_t id, SelectionPoint &point) const {
            for(auto ring : targets) {
                for(auto target : ring) {
                    if(target.globalId == id) {
                        point = target;
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        void Disable(const SelectionPoint &point) {
            targets[point.ringId][point.localId].enabled = false;
        }
        
        double FindClosestPoint(const Mat &extrinscs, SelectionPoint &point, const int ringId = -1) const {
            double bestDist = -1;
            Mat eInv = extrinscs.inv();
            Mat zCorr;
            CreateRotationZ(M_PI / 2, zCorr);
            
            for(auto ring : targets) {
                if(ring.size() == 0 || (ringId != -1 && ringId != (int)ring[0].ringId))
                    continue;
                
                for(auto target : ring) {
                    if(!target.enabled)
                        continue;
                    double dist = GetAngleOfRotation(eInv * target.extrinsics);
                    if (dist < bestDist || bestDist < 0) {
                        point = target;
                        bestDist = dist;
                    }
                }
            }
            
            return bestDist;
        }
        
        uint32_t Size() {
            uint32_t size = 0;
            
            for(auto ring : targets)
                size += ring.size();
            
            return size;
        }
    };
}
#endif
