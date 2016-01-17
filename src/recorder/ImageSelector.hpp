#include <opencv2/opencv.hpp>

#include "../common/image.hpp"
#include "../common/static_counter.hpp"
#include "../math/support.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_IMAGE_SELECTOR_HEADER
#define OPTONAUT_IMAGE_SELECTOR_HEADER

namespace optonaut {
    class ImageSelector {
        private: 
            typedef std::function<void(SelectionInfo)> MatchCallback;

            RecorderGraph &graph;
            MatchCallback callback;

            int currentRing;
            bool isFinished;
            
            double tolerance; 
            bool strictOrder;

            SelectionInfo current;

            void MoveToNextRing() {

                int ringCount = (int)graph.ringCount;
                int newRing = GetNextRing(); 
                
                Assert(newRing >= 0 && newRing < ringCount);
                
                currentRing = newRing;

                current.isValid = false;
            }
        
            int GetNextRing() {
                // Moves from center outward, toggling between top and bottom, 
                // top ring comes before bottom ring.
                int ringCount = (int)graph.ringCount;
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

        public:
            ImageSelector(RecorderGraph &graph, 
                    MatchCallback onNewMatch,
                    double tolerance = M_PI / 16,
                    bool strictOrder = true) :
                graph(graph), callback(onNewMatch), 
                isFinished(false), tolerance(tolerance),
                strictOrder(false) { 
                    
                if(strictOrder) {
                    currentRing = (int)(graph.ringCount - 1) / 2;
                } else {
                    currentRing = -1;
                }
            }

            SelectionInfo GetCurrent() {
                return current;
            }

            void Flush() {
                if(current.isValid) {
                    callback(current);
                }
            }
        
            void Push(const InputImageP image) {

                SelectionPoint next;
                double dist = graph.FindClosestPoint(image->adjustedExtrinsics, 
                        next, currentRing);

                if(dist > tolerance) 
                    return;

                if(!current.isValid) {
                    // Init case - remember this point. 
                    current.closestPoint = next;
                    current.image = image;
                    current.dist = dist;
                    current.isValid = true;
                } else {
                    if(next.globalId == current.closestPoint.globalId) {
                        if(dist < current.dist) {
                            // Better match.
                            current.dist = dist;
                            current.closestPoint = next;
                            current.image = image;
                            current.isValid = true;
                        }
                    } else {
                        // New Match
                        SelectionPoint realNext;
                        graph.GetNextForRecording(current.closestPoint, realNext);

                        if(strictOrder) {
                            // Strict order - refuse to take anything out of order.
                            if(realNext.globalId == next.globalId) {
                                // Valid edge, notify and advance.
                                // Else, ignore. 
                                graph.MarkEdgeAsRecorded(current.closestPoint, next);

                                callback(current);

                                current.isValid = false;

                                if(graph.GetNextForRecording(next, realNext)) {
                                    current.closestPoint = next;
                                    current.dist = dist;
                                    current.image = image;
                                    current.isValid = true;
                                } else {
                                    int nextRing = GetNextRing();
                                    if(nextRing < 0 
                                            || nextRing >= (int)graph.ringCount) {
                                        AssertWM(!isFinished, "Already finished");
                                        isFinished = true;
                                    } else {
                                        MoveToNextRing();
                                    }
                                }
                            }
                        } else {
                            callback(current);
                            current.closestPoint = next;
                            current.dist = dist;
                            current.image = image;
                            current.isValid = true;
                        }
                    } 
                }
            }
    };
}

#endif
