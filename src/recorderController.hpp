#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>

#include "image.hpp"
#include "support.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_RECORDER_CONTROLLER_HEADER
#define OPTONAUT_RECORDER_CONTROLLER_HEADER

namespace optonaut {

    struct SelectionInfo {
        SelectionPoint closestPoint;
        ImageP image;
        double dist;
        bool isValid;
        
        SelectionInfo() : dist(0), isValid(false) {
            
        }
    };

    class RecorderController {
    private:
        RecorderGraph &graph;
        int currentRing;
        
        bool isFinished;
        bool isInitialized;
        
        Mat ballPosition;
        SelectionPoint next;
        SelectionPoint current;
        bool currentDone;
        double bestDist;
        
        double error;
        Mat errorVec;
        
        //Tolerance, measured on sphere, for hits.
        //We sould calc this from buffer, overlap, fov
        const double tolerance = M_PI / 8; //TODO CHANGE FOR RELEASE
        
        void MoveToNextRing(const Mat &cur) {
            
            int ringCount = (int)graph.targets.size();
            int newRing = GetNextRing(); 
            
            assert(newRing >= 0 && newRing < ringCount);
            
            currentRing = newRing;
            MoveToClosestPoint(cur);
        }

        int GetNextRing() {
            //Moves from center outward, toggling between top and bottom, top ring comes before bottom ring.
            int ringCount = (int)graph.targets.size();
            int centerRing = ringCount / 2;
            
            int newRing = currentRing - centerRing;
            //If we are on a bottom or the center ring, move outward.
            if(newRing >= 0) {
                newRing++;
            }
            //Switch bottom and top, or vice versa.
            newRing *= -1;
            newRing = newRing + centerRing;

            return newRing;
        }
        
        void MoveToClosestPoint(const Mat &closest) {
            graph.FindClosestPoint(closest, next, currentRing);
            ballPosition = next.extrinsics.clone();
        }

    public:
        RecorderController(RecorderGraph &graph): 
            graph(graph), isFinished(false), isInitialized(false),
            ballPosition(Mat::eye(4, 4, CV_64F)), bestDist(1000), 
            error(-1), errorVec(3, 1, CV_64F) { }
        
        bool IsInitialized() {
            return isInitialized;
        }
        
        void Initialize(const Mat &initPosition) {
            assert(!isInitialized);
            
            currentRing = (int)graph.targets.size() / 2;
            MoveToClosestPoint(initPosition);
            current = next;
            currentDone = false;
            isInitialized = true;
        }
        
        SelectionInfo Push(const ImageP image, bool isIdle) {
            assert(isInitialized);

            SelectionInfo info;
            info.image = image;
            
            double distCurrent = GetAngleOfRotation(image->adjustedExtrinsics, current.extrinsics);
            double distNext = 1000;

            if(current.globalId != next.globalId) {
                distNext = GetAngleOfRotation(image->adjustedExtrinsics, next.extrinsics);
            } else {
                distNext = distCurrent;
            }
            if(distCurrent < distNext || !currentDone) {
                info.dist = distCurrent;
                info.closestPoint = current;
                error = distCurrent;
            } else {
                info.dist = distNext;
                info.closestPoint = next;
                error = distNext;
                bestDist = info.dist;
                if(current.globalId != next.globalId) {
                    current = next;
                    currentDone = false;
                }
            }
            
            if(!currentDone) {
                ballPosition = current.extrinsics.clone();
            } else {
                ballPosition = next.extrinsics.clone();
            }
            ExtractRotationVector(image->adjustedExtrinsics.inv() * current.extrinsics, errorVec);

            if(isIdle)
                return info;
                
            if(info.dist < tolerance && info.dist <= bestDist) {
                bestDist = info.dist;
                info.isValid = true;
                currentDone = true;
            }
           
            //If we are close enough, and we are closert to next than
            //to current (e.g. current is next), go one step forward.  
            if(distNext < tolerance && next.globalId == current.globalId) {
                SelectionPoint newNext;
                if(graph.GetNextForRecording(next, newNext)) {
                    graph.MarkEdgeAsRecorded(next, newNext);
                    next = newNext;
                } else {
                    int nextRing = GetNextRing();
                    if(nextRing < 0 || nextRing >= (int)graph.targets.size()) {
                        cout << "Push after finish warning." << endl;
                    } else {
                        MoveToNextRing(image->adjustedExtrinsics);
                    }
                }
            }
            
            return info;
        }
        
        bool IsFinished() const {
            return isFinished;
        }
        
        const Mat &GetBallPosition() const {
            return ballPosition;
        }
        
        //Double: All axis, rVec: Single axis.
        double GetError() const {
            return error;
        }
        
        const Mat &GetErrorVector() const {
            return errorVec;
        }
        
        const SelectionInfo FindClosestPoint(const ImageP &image, const int ringId = -1) const {
            assert(isInitialized);
            
            SelectionInfo info;

            info.dist = graph.FindClosestPoint(image->adjustedExtrinsics, info.closestPoint, ringId);
            info.image = image;
            info.isValid = true;
            
            return info;
        }
        
    };
}
#endif
