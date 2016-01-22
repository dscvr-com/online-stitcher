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
    
    struct SelectionInfo {
        SelectionPoint closestPoint;
        InputImageP image;
        double dist;
        bool isValid;
        
        SelectionInfo() : dist(0), isValid(false) {
            
        }
    };

    class ImageSelector {
        public: 
            typedef std::function<void(SelectionInfo)> MatchCallback;
        private: 
            MatchCallback callback;

            int currentRing;
            bool isFinished;
            
            double tolerance; 
            bool strictOrder;

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

        protected: 
            RecorderGraph &graph;
            SelectionInfo current;

            virtual void SetCurrent(const SelectionPoint &closestPoint,
                                    const InputImageP image, 
                                    const double dist) {
                    current.closestPoint = closestPoint;
                    current.image = image;
                    current.dist = dist;
                    current.isValid = true;
            }

            virtual void Invalidate() {
                current.isValid = false;
            }

        public:

            ImageSelector(RecorderGraph &graph, 
                    MatchCallback onNewMatch,
                    double tolerance = M_PI / 16,
                    bool strictOrder = true) :
                callback(onNewMatch), 
                isFinished(false), tolerance(tolerance),
                strictOrder(strictOrder), graph(graph) { 
                    
                if(strictOrder) {
                    currentRing = (int)(graph.ringCount - 1) / 2;
                } else {
                    currentRing = -1;
                }
            }

            SelectionInfo GetCurrent() {
                return current;
            }

            virtual void Flush() {
                if(current.isValid) {
                    callback(current);
                }
            }
        
            virtual bool Push(const InputImageP image) {

                SelectionPoint next;
                double dist = graph.FindClosestPoint(image->adjustedExtrinsics, 
                        next, currentRing);

                if(dist > tolerance) 
                    return false;

                if(!current.isValid) {
                    // Init case - remember this point. 
                    SetCurrent(next, image, dist);
                    return true;
                } else {
                    if(next.globalId == current.closestPoint.globalId) {
                        if(dist < current.dist) {
                            // Better match.
                            SetCurrent(next, image, dist);
                            return true;
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

                                Invalidate();

                                if(graph.GetNextForRecording(next, realNext)) {
                                    SetCurrent(next, image, dist);
                                    return true;
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
                            SetCurrent(next, image, dist);
                            return true;
                        }
                    } 
                }
                return false;
            }
    };

    // An image selector that's got balls.
    class FeedbackImageSelector : public ImageSelector {
        private:
            SelectionPoint ballTarget;
            Mat ballPosition;
            bool hasStarted;
            const int ballLead = 4;
            //const double ballSpeed = M_PI / 128;

            Mat errorVec;
            double error;

            SelectionPoint GetNext(const SelectionPoint &current, 
                                   const bool allowRingSwitch,
                                   const int depth) {
                if(depth == 0)
                    return current;

                SelectionPoint next;

                if(graph.GetNextForRecording(current, next)) {
                    if(current.ringId != next.ringId) {
                        if(allowRingSwitch) {
                            return next;
                        } else {
                            return current;
                        }
                    }

                    return GetNext(next, false, depth - 1); 
                } else {
                    return current;
                }
                
            }

        protected:
            virtual void SetCurrent(const SelectionPoint &closestPoint,
                                    const InputImageP image, 
                                    const double dist) {
                ImageSelector::SetCurrent(closestPoint, image, dist);

                ballTarget = GetNext(closestPoint, true, ballLead);

            }

            virtual void Invalidate() {
                ImageSelector::Invalidate();
            }

        public: 
            FeedbackImageSelector(RecorderGraph &graph, 
                    MatchCallback onNewMatch,
                    double tolerance = M_PI / 16,
                    bool strictOrder = true) : 
                ImageSelector(graph, onNewMatch, tolerance, strictOrder), 
                hasStarted(false) {
                
            }

            using ImageSelector::Push;

            bool Push(InputImageP image, bool isIdle) {

                if(!hasStarted) {
                    ballPosition = image->originalExtrinsics;
                } else {
                    // TODO: LERP!
                }
                
                if(isIdle) {
                    return false;
                }

                hasStarted = true;
                
                bool result = ImageSelector::Push(image);
                
                ExtractRotationVector(
                        image->adjustedExtrinsics.inv() * ballPosition, errorVec);

                error = GetAngleOfRotation(image->adjustedExtrinsics, ballPosition);

                return result;
            }

            const Mat &GetBallPosition() const {
                return ballPosition;
            }
            
            double GetError() const {
                return error;
            }
            
            const Mat &GetErrorVector() const {
                return errorVec;
            }
    };
}

#endif
