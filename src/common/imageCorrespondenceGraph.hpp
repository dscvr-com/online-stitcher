#include <opencv2/opencv.hpp>
#include <mutex>

#include "graph.hpp"

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
            ImageCorrespondenceGraph() { }

            ValueType Register(Mat &a, size_t aId, Mat &b, size_t bId) {
                ValueType aToB;
                ValueType bToA;

                auto res = GetCorrespondence(a, aId, b, bId, aToB, bToA);
               
                {
                    unique_lock<mutex> lock(graphLock); 
                    relations.Insert(aId, bId, aToB);
                    relations.Insert(bId, aId, bToA);
                }
                
                return res;
            }

            virtual ValueType GetCorrespondence(const Mat &a, size_t aId, const Mat &b, size_t bId, ValueType &aToB, ValueType &bToA) = 0;

    };

}

#endif
