#include <opencv2/opencv.hpp>
#include <mutex>

#include "graph.hpp"
#include "../io/inputImage.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_IMAGE_CORRESPONDENCE_GRAPH_HEADER
#define OPTONAUT_IMAGE_CORRESPONDENCE_GRAPH_HEADER

namespace optonaut {

    /*
     * A graph that holds correspondences between images. 
     */
    template <typename ValueType>
    class ImageCorrespondenceGraph {
        protected: 
            SparseGraph<ValueType> relations;
            mutex graphLock;
            static const bool debug = true;

        public:
            /*
             * Edge type. 
             */
            typedef typename SparseGraph<ValueType>::Edge Edge;
            /*
             * Edge collection type.
             */
            typedef typename SparseGraph<ValueType>::Edges Edges;
            /*
             * Adjacency list type. 
             */
            typedef typename SparseGraph<ValueType>::AdjList AdjList;

            /*
             * Creates a new empty graph. 
             */
            ImageCorrespondenceGraph() { }

            /*
             * Registers the given images with this specific implementation. 
             *
             * For getting the bi-directional correspondence between the images, the implementation 
             * of the subclass is used. 
             *
             * @param a The first image.
             * @param b The second image. 
             *
             * @returns The correspondence from a to b. 
             */
            ValueType Register(InputImageP a, InputImageP b) {
                ValueType aToB;
                ValueType bToA;

                auto res = GetCorrespondence(a, b, aToB, bToA);
               
                InsertCorrespondence(a->id, b->id, aToB, bToA);
                
                return res;
            }

            /*
             * Inserts a correspondence between the two given images 
             * with the given value. This method is thread safe. 
             * 
             * @param aId Id of the first image. 
             * @param bId Id of the second image. 
             * @param aToB Correspondence from a to b.
             * @param bToA Correspondence from b to a.
             */
            void InsertCorrespondence(int aId, int bId, 
                    const ValueType &aToB, const ValueType &bToA) {
                {
                    unique_lock<mutex> lock(graphLock); 
                    relations.Insert(aId, bId, aToB);
                    relations.Insert(bId, aId, bToA);
                }
            }

            /*
             * @returns All the correspondences in this graph. 
             */
            const AdjList &GetEdges() const {
                return relations.GetEdges();
            }

            bool GetEdge(const int aId, const int bId, Edge &edge) {
                return relations.GetEdge(aId, bId, edge);
            }
            
            /*
             * Prints all correspondences, for testing. 
             */
            void PrintCorrespondence() {
                for(auto &adj : relations.GetEdges()) {
                    for(auto &edge : adj.second) {
                        cout << edge.from << " -> " << edge.to << ": " << edge.value << endl;
                    }
                }
            }

            /*
             * Abstract method to get the bi-directional correspondence betwenn two images. To be implemented by subclass. 
             */
            virtual ValueType GetCorrespondence(InputImageP a, InputImageP b, ValueType &aToB, ValueType &bToA) = 0;
    };

}

#endif
