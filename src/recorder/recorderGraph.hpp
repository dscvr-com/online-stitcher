#include <opencv2/opencv.hpp>
#include <math.h>
#include <list>
#include <vector>

#include "../io/inputImage.hpp"
#include "../common/image.hpp"
#include "../common/bimap.hpp"
#include "../math/support.hpp"

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
        uint32_t ringSize;
        double hPos;
        double vPos;
        double hFov;
        double vFov;
        Mat extrinsics;
        
        SelectionPoint() : globalId(0),
            localId(0), ringId(0), ringSize(0), hPos(0), vPos(0), hFov(0), 
            vFov(0), extrinsics(0, 0, CV_64F) {
        }
    };
    
    struct SelectionEdge {
        uint32_t from;
        uint32_t to;
        bool recorded;
        
        SelectionEdge() : from(0), to(0), recorded(false) {
            
        }
    };
    
    class RecorderGraph {
    private: 
        vector<vector<SelectionEdge>> adj;
        vector<vector<SelectionPoint>> targets;
        vector<SelectionPoint*> targetsById;
    public:
        
        static const int ModeAll = 0;   // 5 rings
        static const int ModeCenter = 1;  // 1 ring
        static const int ModeTruncated = 2; // 3 ring
        static const int ModeNoBot = 3; // 3 ring only the top
        static const int ModeTinyDebug = 1337;
        
        static constexpr float DensityHalf = 0.5;
        static constexpr float DensityNormal = 1;
        static constexpr float DensityDouble = 2;
        static constexpr float DensityQadruple = 4;
        
        const Mat intrinsics;
        const uint32_t ringCount;

        RecorderGraph(uint32_t ringCount, const Mat &intrinsics)
            : intrinsics(intrinsics), ringCount(ringCount) {
            targets.reserve(ringCount); 
        }
        
        RecorderGraph(const RecorderGraph &c)
            : adj(c.adj), targets(c.targets), 
            intrinsics(c.intrinsics), ringCount(c.ringCount)
        {
            for(auto ring : targets) {
                for(auto target : ring) {
                    AssertEQ(target.globalId, (uint32_t)targetsById.size());
                    targetsById.push_back(&target);
                }
            }
        }

        uint32_t AddNewRing(uint32_t ringSize) {
            AssertGT((size_t)ringCount, targets.size()); 
            targets.push_back(vector<SelectionPoint>(ringSize));

            return (uint32_t)targets.size() - 1;
        }

        void AddNewNode(const SelectionPoint &point) {
            //Check for index consistency. 
            AssertGT(targets.size(), (size_t)point.ringId); 
            AssertGT(targets[point.ringId].size(), (size_t)point.localId);
            while(targetsById.size() <= point.globalId) {
                targetsById.push_back(NULL);
            }

            targets[point.ringId][point.localId] = point;
            targetsById[point.globalId] = &(targets[point.ringId][point.localId]);
        } 

        void AddEdge(const SelectionEdge &edge) {
            while(adj.size() <= edge.from) {
                adj.push_back(vector<SelectionEdge>());
            }
            adj[edge.from].push_back(edge);
        }
       
        const vector<vector<SelectionPoint>> &GetRings() const {
            return targets;
        }
        
        const vector<SelectionPoint*> &GetTargetsById() const {
            return targetsById;
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

        bool GetNext(const SelectionPoint &current, SelectionPoint &next) const {
            auto it = adj[current.globalId].begin();

            if(it == adj[current.globalId].end()) {
                return false;
            } else {
                return GetPointById(it->to, next);
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
            
            if(id < targetsById.size()) {
                point = *(targetsById[id]);
                return true;
            }
            
            return false;
        }
        
        double FindClosestPoint(const Mat &extrinscs, SelectionPoint &point, const int ringId = -1) const {
            double bestDist = -1;
            Mat eInv = extrinscs.inv();
            
            for(auto &ring : targets) {
                if(ring.size() == 0 || 
                        (ringId != -1 && ringId != (int)ring[0].ringId)) {
                    continue;
                }
                
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

        vector<InputImageP> SelectBestMatches(const vector<InputImageP> &imgs) {
            BiMap<size_t, uint32_t> dummy;
            return SelectBestMatches(imgs, dummy); 
        }

        vector<InputImageP> SelectBestMatches(const vector<InputImageP> &_imgs, 
                BiMap<size_t, uint32_t> &imagesToTargets) const {

            const double thresh = M_PI / 16;

            std::list<InputImageP> imgs(_imgs.begin(), _imgs.end());
            vector<InputImageP> res;

            for(auto &ring : targets) {
                for(size_t i = 0; i < ring.size(); i++) {
                    auto target = ring[i];
                    auto compare = [&target](
                            const InputImageP &a, 
                            const InputImageP &b) {
                        auto dA = GetAngleOfRotation(a->adjustedExtrinsics.inv() * 
                                target.extrinsics);
                        auto dB = GetAngleOfRotation(b->adjustedExtrinsics.inv() * 
                                target.extrinsics);

                        return dA < dB;
                    };

                    //Todo - keep track of max distance!
                    auto it = std::min_element(imgs.begin(), imgs.end(), compare);
                    InputImageP min = *it;

                    if(it == imgs.end())
                        continue;
                    
                    double dist = GetAngleOfRotation(target.extrinsics,
                            min->adjustedExtrinsics);

                    if(abs(dist) > thresh)
                        continue;


                    imgs.erase(it);
                    imagesToTargets.Insert(min->id, target.globalId);

                    res.push_back(min);
                }
            }

            return res;
        }
        
        vector<vector<InputImageP>> SplitIntoRings(const vector<InputImageP> &imgs) const {
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
