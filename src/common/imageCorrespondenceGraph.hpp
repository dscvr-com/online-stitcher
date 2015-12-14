#include <opencv2/opencv.hpp>
#include <mutex>

#include "graph.hpp"
#include "../io/inputImage.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_IMAGE_CORRESPONDENCE_GRAPH_HEADER
#define OPTONAUT_IMAGE_CORRESPONDENCE_GRAPH_HEADER

namespace optonaut {

    template <typename ValueType>
    class ImageCorrespondenceGraph {
        protected: 
            SparseGraph<ValueType> relations;
            mutex graphLock;
            static const bool debug = true;

        public:
            typedef typename SparseGraph<ValueType>::Edge Edge;
            typedef typename SparseGraph<ValueType>::Edges Edges;
            typedef typename SparseGraph<ValueType>::AdjList AdjList;
            ImageCorrespondenceGraph() { }

            ValueType Register(InputImageP a, InputImageP b) {
                ValueType aToB;
                ValueType bToA;

                auto res = GetCorrespondence(a, b, aToB, bToA);
               
                {
                    unique_lock<mutex> lock(graphLock); 
                    relations.Insert(a->id, b->id, aToB);
                    relations.Insert(b->id, a->id, bToA);
                }
                
                return res;
            }

            const AdjList &GetEdges() const {
                return relations.GetEdges();
            }
            
            void PrintCorrespondence() {
                for(auto &adj : relations.GetEdges()) {
                    for(auto &edge : adj.second) {
                        cout << edge.from << " -> " << edge.to << ": " << edge.value << endl;
                    }
                }
            }


            virtual ValueType GetCorrespondence(InputImageP a, InputImageP b, ValueType &aToB, ValueType &bToA) = 0;

    };

}

#endif
