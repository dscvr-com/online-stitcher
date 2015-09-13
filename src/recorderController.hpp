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
        SelectionPoint ballTarget;
        SelectionPoint prevTarget;
        double prevDist;
        
        double error;
        Mat errorVec;
        
        //Tolerance, measured on sphere, for hits.
        //We sould calc this from buffer, overlap, fov
        const double tolerance = M_PI / 50;
        
        //Ball Speed per second, in radians
        const double ballSpeed = M_PI / 2;
        
        size_t lt;
        
        
        void MoveToNextRing(const Mat &cur) {
            
            //Moves from center outward, toggling between top and bottom, top ring comes before bottom ring.
            int ringCount = (int)graph.targets.size();
            int centerRing = ringCount / 2;
            
            currentRing = currentRing - centerRing;
            //If we are on a bottom or the center ring, move outward.
            if(currentRing >= 0) {
                currentRing++;
            }
            //Switch bottom and top, or vice versa.
            currentRing *= -1;
            currentRing = currentRing + centerRing;
            
            assert(currentRing >= 0 && currentRing < ringCount);
            
            MoveToClosestPoint(cur);
        }
        
        void MoveToClosestPoint(const Mat &closest) {
            graph.FindClosestPoint(closest, ballTarget, currentRing);
            ballPosition = ballTarget.extrinsics.clone();
        }

    public:
        RecorderController(RecorderGraph &graph): graph(graph), isFinished(false), isInitialized(false),
            ballPosition(Mat::eye(4, 4, CV_64F)), prevDist(100), error(-1), errorVec(3, 1, CV_64F), lt(0) {
        }
        
        bool IsInitialized() {
            return isInitialized;
        }
        
        void Initialize(const Mat &initPosition) {
            assert(!isInitialized);
            
            currentRing = (int)graph.targets.size() / 2;
            MoveToClosestPoint(initPosition);
            isInitialized = true;
        }
        
        SelectionInfo Push(const ImageP image, bool isIdle) {
            assert(isInitialized);

            double viewDist = GetAngleOfRotation(image->extrinsics, ballTarget.extrinsics);
            size_t t = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
            
            SelectionInfo info;
            info.closestPoint = ballTarget;
            info.image = image;

            if(!isIdle && lt != 0) {
                //If we reached a viewpoint...
                if(viewDist < tolerance) {
                    SelectionPoint next;
                    graph.MarkEdgeAsRecorded(prevTarget, ballTarget);
                    if(graph.GetNextForRecording(ballTarget, next)) {
                        prevTarget = ballTarget;
                        prevDist = 100;
                        ballTarget = next;
                        cout << "Move next" << endl;
                    } else {
                        //Ring switch or finish
                        MoveToNextRing(image->extrinsics);
                        cout << "Ring Jump" << endl;
                    }
                }

                //Animate ball.
                double dist = GetAngleOfRotation(ballPosition, ballTarget.extrinsics);
                //Delta time
                double dt = (double)(t - lt) / 1000.0;
                //Delta pos
                double dp = cos(dist) * ballSpeed * dt;
                Mat newPos(4, 4, CV_64F);
                Lerp(ballPosition, ballTarget.extrinsics, dp, newPos);
                    
                ballPosition = newPos;
              
            }
            
            
            error = GetAngleOfRotation(image->extrinsics, ballPosition);
            ExtractRotationVector(image->extrinsics.inv() * ballPosition, errorVec);
            
            info.dist = viewDist;
            if(info.dist < tolerance && info.dist < prevDist) {
                info.isValid = true;
                prevDist = info.dist;
            }
            
            lt = t;
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

            info.dist = graph.FindClosestPoint(image->extrinsics, info.closestPoint, ringId);
            info.image = image;
            info.isValid = true;
            
            return info;
        }
        
    };
}
#endif
