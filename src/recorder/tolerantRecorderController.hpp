#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>
#include <unordered_set>

#include "../common/image.hpp"
#include "../math/support.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_TOLERANT_RECORDER_CONTROLLER_HEADER
#define OPTONAUT_TOLERANT_RECORDER_CONTROLLER_HEADER

namespace optonaut {

    class TolerantRecorderController {
    private:
        //unordered_set<int> done;
        SelectionInfo best;
        RecorderGraph &graph;
    public:
        const double tolerance = M_PI / 8; //High tolerance, since we have to catch up with alignment

        TolerantRecorderController(RecorderGraph &graph) : graph(graph) {
            best.isValid = false;
        }
       
        //This only works if we can guarantee that the images will come in order
        //(which we can if we do pre-selection). 
        SelectionInfo Push(const InputImageP image) {
            SelectionInfo current;
            current.dist = graph.FindClosestPoint(image->adjustedExtrinsics, current.closestPoint);
            current.image = image;
            
            //Constraint at least a little. 
            if(current.dist > tolerance) {
                return best;    
            }

            //Avoid duplicates.
            //if(done.find(current.closestPoint.globalId) != done.end()) {
            //    return best;
            //} 
            
            current.isValid = true;
             
            if(!best.isValid || 
                    current.closestPoint.globalId != best.closestPoint.globalId ||
                    current.dist < best.dist) {
                //cout << "Tolerant recorder found a better match" << endl;
                //done.insert(best.closestPoint.globalId);
                best = current;
            }

            return best;
        }
    };
}
#endif
