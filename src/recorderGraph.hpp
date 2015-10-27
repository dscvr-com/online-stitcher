
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
        double hPos;
        double vPos;
        double hFov;
        double vFov;
        Mat extrinsics;
        
        SelectionPoint() : globalId(0),
            localId(0), ringId(0), hPos(0), vPos(0), hFov(0), vFov(0) {
        }
    };
    
    struct SelectionEdge {
        uint32_t from;
        uint32_t to;
        bool recorded;
        
        SelectionEdge() : from(0), to(0), recorded(false) {
            
        }
    };
    
    //Todo - Declare Friend with recorder Graph Generator, then make props private
    class RecorderGraph {
    public:
        
        static const int ModeAll = 0;
        static const int ModeCenter = 1;
        static const int ModeTruncated = 2;
        static const int ModeNoBot = 3;
        static const int ModeTinyDebug = 1337;
        
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
        
        bool GetEdge(const SelectionPoint& left, const SelectionPoint& right, SelectionEdge &outEdge) const {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    outEdge = *it;
                    return true;
                }
            }
            return false;
        }
        
        void MarkEdgeAsRecorded(const SelectionPoint& left, const SelectionPoint& right) {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    it->recorded = true;
                }
            }
        }
        
        bool GetNextForRecording(const SelectionPoint &current, SelectionPoint &next) const {
            auto it = adj[current.globalId].begin();
            
            while(adj[current.globalId].end() != it)
            {
                if(!it->recorded) {
                    return GetPointById(it->to, next);
                }
                it++;
            }
            
            return false;
        }
        
        bool GetPointById(uint32_t id, SelectionPoint &point) const {
            for(auto &ring : targets) {
                for(auto target : ring) {
                    if(target.globalId == id) {
                        point = target;
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        double FindClosestPoint(const Mat &extrinscs, SelectionPoint &point, const int ringId = -1) const {
            double bestDist = -1;
            Mat eInv = extrinscs.inv();
            
            for(auto &ring : targets) {
                if(ring.size() == 0 || (ringId != -1 && ringId != (int)ring[0].ringId))
                    continue;
                
                for(auto &target : ring) {
                    double dist = GetAngleOfRotation(eInv * target.extrinsics);
                    if (dist < bestDist || bestDist < 0) {
                        point = target;
                        bestDist = dist;
                    }
                }
            }
            
            return bestDist;
        }

        int FindAssociatedRing(const Mat &extrinsics, const double tolerance = M_PI / 8) const {
            assert(targets.size() > 0);

            SelectionPoint pt;
            if(FindClosestPoint(extrinsics, pt) <= tolerance) {
                return pt.ringId;
            }

            return -1;
        }

        int GetChildRing(int ring) {
            int c = (int)(targets.size() - 1) / 2;

            if(ring == c) {
                return ring;
            } else if(ring > c) {
                return ring + 1;
            } else {
                return ring - 1;
            }
        }

        bool HasChildRing(int ring) {
            int c = GetChildRing(ring);
            if(c < 0 || c >= (int)targets.size()) {
                return false;
            } else {
                return targets[c].size() > 0;
            }
        }

        int GetParentRing(int ring) {
            int c = (int)(targets.size() - 1) / 2;

            if(ring == c) {
                return ring;
            } else if(ring > c) {
                return ring - 1;
            } else {
                return ring + 1;
            }
        }
        
        uint32_t Size() {
            uint32_t size = 0;
            
            for(auto &ring : targets)
                size += ring.size();
            
            return size;
        }
        
        
        vector<vector<InputImageP>> SplitIntoRings(vector<InputImageP> &imgs) const {
            vector<vector<InputImageP>> rings(this->GetRings().size());
            
            for(auto img : imgs) {
                int r = this->FindAssociatedRing(img->originalExtrinsics);
                if(r == -1)
                    continue;
                rings[r].push_back(img);
            }
            
            return rings;
        }
    };
}
#endif
