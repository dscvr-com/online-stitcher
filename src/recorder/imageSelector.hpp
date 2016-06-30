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
   
    /*
     * Represents a match of an image and a corresponding selection point. 
     */
    struct SelectionInfo {
        SelectionPoint closestPoint; // The associated point
        InputImageP image; // The associated image
        double dist; // The distance, in radians on a sphere
        bool isValid; // Is the match valid? 
        
        SelectionInfo() : dist(0), isValid(false) {
            
        }
    };

    /*
     * Class responsible of matching series of images with selection
     * points in a recorder graph in order. 
     *
     * This class basically works by holding a reference to the current selection
     * point. For each input image, we move our current selection point. We assume a best match
     * for a selection point if we get a new image that is not a better match than the previous one.  
     * If this case happens, the callback is invoked. 
     */
    class ImageSelector {
        public: 
            typedef std::function<void(SelectionInfo)> MatchCallback;
        private: 
            MatchCallback callback;

            bool isFinished;
            
            Vec3d tolerance;
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
        
            bool CheckIfWithinTolerance(const Mat &a, const Mat &b) {
                Mat rvec;
                ExtractRotationVector(a.inv() *  b, rvec);
                
                int ringCount = (int)graph.ringCount;
                
                if(currentRing == (ringCount) / 2){
                    // Currently working on center ring - normal tolerance. 
                    for(int i = 0; i < 3; i++) {
                        if(abs(rvec.at<double>(i)) > tolerance(i)) {
                            return false;
                        }
                    }

                } else {
                    // Working on outer rings - extended tolerance. 
                    for(int i = 0; i < 3; i++) {
                        if (i == 2){
                            if(abs(rvec.at<double>(i)) > (tolerance(i)* 1.5)) {
                                return false;
                            }
                        }
                        else{
                            if(abs(rvec.at<double>(i)) > tolerance(i)) {
                                return false;
                            }
                        }
                    }
                    
                }
                
                return true;
           }

        protected:
            RecorderGraph &graph;
            SelectionInfo current;
            int currentRing;

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

            /*
             * Creates a new ImageSelector based on the given graph and the given callback.
             *
             * @param graph The recorder graph to use. 
             * @param onNewMatch The callback onNewMatch is called whenever a new best match is found for
             * a selection point. 
             * @param tolerance The rotational tolerance, for each axis. 
             * @param strictOrder If true, all selection points must be visited in order. 
             *
             */
            ImageSelector(RecorderGraph &graph, 
                    MatchCallback onNewMatch,
                    Vec3d tolerance,
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

            /*
             * Returns the current match. 
             */
            SelectionInfo GetCurrent() const {
                return current;
            }

            /*
             * Notifies the selector that processing is finished. 
             */
            virtual void Flush() {
                if(current.isValid) {
                    // It is safe to assume that the last match, if it was valid
                    // was also the best one. 
                    callback(current);
                }
            }
           
            /*
             * Returns true if all selection points have been visited. 
             */ 
            bool IsFinished() {
                return isFinished;
            }
        
            /*
             * Pushes an image to this selector and advances the internal state. 
             */
            virtual bool Push(const InputImageP image) {

                SelectionPoint next;
                double dist = graph.FindClosestPoint(image->adjustedExtrinsics, 
                        next, currentRing);
                
                if(!CheckIfWithinTolerance(image->adjustedExtrinsics, next.extrinsics)) {
                    return false;
                }

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
                        if(!graph.GetNextForRecording(
                                    current.closestPoint, 
                                    realNext)) {
                            // We're already finished. There is no next. 
                            return false;
                        }

                        if(strictOrder) {
                            // Strict order - refuse to take anything out of order.
                            if(realNext.globalId != next.globalId) {
                                if(CheckIfWithinTolerance(image->adjustedExtrinsics, 
                                            realNext.extrinsics)) {
                                    next = realNext;
                                }
                            }
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
                            } else {
                                //cout << "Reject Unordered" << endl;
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

    /*
     * An extension of ImageSelector that calculates guidenceinformation to show in the UI. 
     */
    class FeedbackImageSelector : public ImageSelector {
        private:
            SelectionPoint ballTarget;
            Mat ballPosition;
            bool hasStarted;
            const int ballLead = 2;

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
        
            void UpdateBallPosition(const Mat &newPosition) {
                ballPosition = newPosition;
                
                Slerp(ballPosition, newPosition, 0.5, ballPosition);
            }

        protected:
            virtual void SetCurrent(const SelectionPoint &closestPoint,
                                    const InputImageP image, 
                                    const double dist) {
                ImageSelector::SetCurrent(closestPoint, image, dist);
                ballTarget = GetNext(current.closestPoint, true, ballLead);
            }

            virtual void Invalidate() {
                ImageSelector::Invalidate();
            }

        public: 
            FeedbackImageSelector(RecorderGraph &graph, 
                    MatchCallback onNewMatch,
                    Vec3d tolerance) :
                ImageSelector(graph, onNewMatch, tolerance, true),
                hasStarted(false), errorVec(Mat::zeros(3, 1, CV_64F)), error(0) {
            
            }
        
            using ImageSelector::Push;

            /*
             * Pushes an input image to this instance and advances the internal state. 
             */
            bool Push(InputImageP image, bool isIdle) {

                if(!hasStarted || !current.isValid) {
                    // Make suggested initial position dependet on vertical position only for smoothness. 
                    graph.FindClosestPoint(image->adjustedExtrinsics,
                                                         ballTarget, currentRing);
                    
                    Mat rDiff;
                    ExtractRotationVector(image->adjustedExtrinsics * ballTarget.extrinsics.inv(), rDiff);
                    Mat mDiff;
                    CreateRotationY(rDiff.at<double>(1), mDiff);
                    
                    UpdateBallPosition(mDiff * ballTarget.extrinsics);
                    
                } else {
                    // Do noremal movement, also update the ball position if applicable. 
                    if(isIdle) {
                        ballTarget = GetNext(current.closestPoint, true, 1);
                        UpdateBallPosition(ballTarget.extrinsics);
                    } else {
                        UpdateBallPosition(ballTarget.extrinsics);
                    }
                }
                
                // Extract error/guidance information
                ExtractRotationVector(
                    image->adjustedExtrinsics.inv() * ballPosition, errorVec);
                
                error = GetAngleOfRotation(image->adjustedExtrinsics, ballPosition);

                if(isIdle) {
                    return false;
                }

                hasStarted = true;
                
                return ImageSelector::Push(image);
            }

            /*
             * Gets the guidance ball position (in other words, the position of the next keyframe)
             * as rotation matrix. 
             */
            const Mat &GetBallPosition() const {
                return ballPosition;
            }
           
            /*
             * Gets the absolute guidance error in radians. 
             */ 
            double GetError() const {
                return error;
            }

            /*
             * Gets the guidance error for all axis. 
             */
            const Mat &GetErrorVector() const {
                return errorVec;
            }
    };
}

#endif
