#include <opencv2/opencv.hpp>
#include <math.h>
#include <list>
#include <vector>

#include "../io/inputImage.hpp"
#include "../io/io.hpp"
#include "../common/image.hpp"
#include "../common/bimap.hpp"
#include "../math/support.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_RECORDER_GRAPH_HEADER
#define OPTONAUT_RECORDER_GRAPH_HEADER

namespace optonaut {

    /*
     * Represents a node in the recorder graph. This is a point 
     * where at least one image needs to be captured while recording. 
     */     
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
   
    /*
     * Represents an edge in the recorder graph. 
     * Usually, all edges in a ring need to be traversed during recording. 
     */ 
    struct SelectionEdge {
        uint32_t from;
        uint32_t to;
        bool recorded;
        
        SelectionEdge() : from(0), to(0), recorded(false) {
            
        }
    };
   
    /*
     * This class defines the graph that has to be traversed during recording. 
     * It defines the position and order of all images that should 
     * be taken while recording an optograph. 
     */ 
    class RecorderGraph {
    private: 
        vector<vector<SelectionEdge>> adj;
        vector<vector<SelectionPoint>> targets;
        vector<SelectionPoint*> targetsById;
    public:
        // Recorder graph modes, usually used while generating. 
        static constexpr int ModeAll = 0; // Full sphere
        static constexpr int ModeCenter = 1; // Only center ring
        static constexpr int ModeTruncated = 2; // Omit top and bottom ring (usually leads three rings)
        static constexpr int ModeNoBot = 3; // Omits bottom ring
        static constexpr int ModeTinyDebug = 1337; // Three ring slices. Good for debugging state transistions during recording.  
       
        // Some predefined densities.  
        static constexpr float DensityHalf = 0.5;
        static constexpr float DensityNormal = 1;
        static constexpr float DensityDouble = 2;
        static constexpr float DensityQadruple = 4;
        
        const Mat intrinsics;
        const uint32_t ringCount;

        /*
         * Creates a new, empty, recorder graph. 
         */
        RecorderGraph(uint32_t ringCount, const Mat &intrinsics)
            : intrinsics(intrinsics), ringCount(ringCount) {
            targets.reserve(ringCount); 
        }
       
        /*
         * Copy constructor.  
         */
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

        /* 
         * Adds a new ring to the recorder graph. 
         */
        uint32_t AddNewRing(uint32_t ringSize) {
            AssertGT((size_t)ringCount, targets.size()); 
            targets.push_back(vector<SelectionPoint>(ringSize));

            return (uint32_t)targets.size() - 1;
        }

        /*
         * Adds a new node to the recorder graph. 
         */
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

        /*
         * Adds a new edge to the recorder graph. 
         */
        void AddEdge(const SelectionEdge &edge) {
            while(adj.size() <= edge.from) {
                adj.push_back(vector<SelectionEdge>());
            }
            adj[edge.from].push_back(edge);
        }
      
        /* 
         * Gets all rings from this recorder graph.  
         */
        const vector<vector<SelectionPoint>> &GetRings() const {
            return targets;
        }
        
        /*
         * Gets a lookup table from selection point Ids to selection points. 
         */
        const vector<SelectionPoint*> &GetTargetsById() const {
            return targetsById;
        }
       
        /*
         * Removes an edge from this recorder graph.  
         */
        void RemoveEdge(const SelectionPoint& left, const SelectionPoint& right) {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    adj[left.globalId].erase(it);
                    break;
                }
            }
        }
        
        /*
         * Tries to get an edge between two given points. Returns true if the edge was found. 
         */
        bool GetEdge(const SelectionPoint& left, const SelectionPoint& right, SelectionEdge &outEdge) const {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    outEdge = *it;
                    return true;
                }
            }
            return false;
        }
       
        /*
         * Marks the given edge as recorded.
         */ 
        void MarkEdgeAsRecorded(const SelectionPoint& left, const SelectionPoint& right) {
            for(auto it = adj[left.globalId].begin(); it != adj[left.globalId].end(); it++) {
                if(it->to == right.globalId) {
                    it->recorded = true;
                }
            }
        }

        /*
         * Gets the next selection point. That is the first point that is connected to 
         * the given selection point. 
         */
        bool GetNext(const SelectionPoint &current, SelectionPoint &next) const {
            auto it = adj[current.globalId].begin();

            if(it == adj[current.globalId].end()) {
                return false;
            } else {
                return GetPointById(it->to, next);
            }
        }
       
       int GetNextRing(const int currentRing) const {
            // Moves from center outward, toggling between top and bottom, 
            // top ring comes before bottom ring.
            int ringCount = (int)this->ringCount;
            int centerRing = (ringCount - 1) / 2;
            
            int newRing = currentRing - centerRing;
            // If we are on a bottom or the center ring, move outward.
            if(newRing <= 0) {
                newRing--;
            }
            // Switch bottom and top, or vice versa.
            newRing *= -1;
            newRing = newRing + centerRing;

            return newRing;
        }
       
        /*
         * Gets the next selection point that has not been recorded so far. That is 
         * the selection point that is connected by an edge that was not traversed during recording.  
         */ 
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
       
        /*
         * Gets a selection point by its id. Returns true if the point was found. 
         */ 
        bool GetPointById(uint32_t id, SelectionPoint &point) const {
            
            if(id < targetsById.size()) {
                point = *(targetsById[id]);
                return true;
            }
            
            return false;
        }
       
        /*
         * Finds the point closest to the given position. Optionally restricts the search to a given ring. 
         *
         * Returns the distance to the found point. 
         */ 
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

        /*
         * Finds the ring closest to the given position. 
         */
        int FindAssociatedRing(const Mat &extrinsics, const double tolerance = M_PI / 8) const {
            Assert(targets.size() > 0);

            SelectionPoint pt;
            if(FindClosestPoint(extrinsics, pt) <= tolerance) {
                return pt.ringId;
            }

            return -1;
        }

        /*
         * Returns the index of the child ring for the given ring. 
         */
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

        /*
         * Checks if the given ring has a child ring. 
         */
        bool HasChildRing(int ring) {
            int c = GetChildRing(ring);
            if(c < 0 || c >= (int)targets.size()) {
                return false;
            } else {
                return targets[c].size() > 0;
            }
        }

        /*
         * Returns the parent ring of the given ring. 
         */
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
       
        /*
         * Returns the count of all selection points in this recorder graph. 
         */ 
        uint32_t Size() const {
            uint32_t size = 0;
            
            for(auto &ring : targets)
                size += ring.size();
            
            return size;
        }

        /*
         * Greedely selects the best matches for the given images. 
         */
        vector<InputImageP> SelectBestMatches(const vector<InputImageP> &imgs, 
                bool allowDuplicates = false) {
            BiMap<size_t, uint32_t> dummy;
            return SelectBestMatches(imgs, dummy, allowDuplicates); 
        }

        /*
         * Greedely selects the best matches for the given images. 
         */
        vector<InputImageP> SelectBestMatches(const vector<InputImageP> &_imgs, 
                BiMap<size_t, uint32_t> &imagesToTargets,
                bool allowDuplicates = false) const {

            const double thresh = M_PI / 16;

            std::list<InputImageP> imgs(_imgs.begin(), _imgs.end());
            vector<InputImageP> res;

            for(auto &ring : targets) {
                for(size_t i = 0; i < ring.size(); i++) {

                    if(imgs.size() == 0)
                        break;

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


                    if(!allowDuplicates) {
                        imgs.erase(it);
                    }
                    imagesToTargets.Insert(min->id, target.globalId);

                    res.push_back(min);
                }
            }

            return res;
        }

        /*
         * For each selection point in this graph, that does not have a match
         * in the given images, creates a dummy image and inserts it to the image vector 
         * and the image map.
         *
         * @param imgs The images to check the points for.
         * @param imagesToTargets Image to points map, as generated by SelectBestMatches.
         * @param color The color the generated images should have. 
         * @param size The size the generated images should have. 
         */
        void AddDummyImages(vector<InputImageP> &imgs, BiMap<size_t, uint32_t> &imagesToTargets, const Scalar &color, const cv::Size &size) const {

            AssertFalseInProduction(true); // Never use this method in production. It uses fixed paths. 

            map<uint32_t, const SelectionPoint*> targets;

            for(auto &ring : this->targets) {
                for(auto &target : ring) {
                    targets[target.globalId] = &target;
                }
            }

            int maxId = 0;

            for(auto &img : imgs) {
                uint32_t id;
                Assert(imagesToTargets.GetValue(img->id, id));
                auto it = targets.find(id);
                if(it != targets.end()) { // In case of duplicate, do not double erase
                    targets.erase(it); 
                }

                if(img->id <= maxId) {
                    maxId = img->id + 1;
                }
            }

            for(auto &pair : targets) {
                auto target = pair.second;

                Mat img(size, CV_8UC3, color);
                InputImageP image = make_shared<InputImage>();
                image->image = Image(img);
                target->extrinsics.copyTo(image->originalExtrinsics);
                target->extrinsics.copyTo(image->adjustedExtrinsics);
                imgs[0]->intrinsics.copyTo(image->intrinsics);
                image->id = maxId++;

                imgs.push_back(image);
                    
                imagesToTargets.Insert(image->id, target->globalId);

                auto path = "tmp/" + ToString(image->id) + ".jpg";
                image->image.source = path;
                InputImageToFile(image, path);
            }
        }
       
        /*
         * Splits all given images into rings, according to this graph. 
         */ 
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
