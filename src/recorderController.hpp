#include <opencv2/opencv.hpp>
#include <math.h>

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
        
        //Ball Speed per frame, in radians
        const double ballSpeed = M_PI / 80;
        
        
        void MoveToNextRing() {
            
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
        }

    public:
        RecorderController(RecorderGraph &graph): graph(graph), isFinished(false), isInitialized(false),
            ballPosition(Mat::eye(4, 4, CV_64F)), prevDist(100), error(-1), errorVec(3, 1, CV_64F) {
    
        }
        
        bool IsInitialized() {
            return isInitialized;
        }
        
        void Initialize(const Mat &initPosition) {
            assert(!isInitialized);
            
            currentRing = (int)graph.targets.size() / 2;

            graph.FindClosestPoint(initPosition, ballTarget, currentRing);
            
            ballPosition = ballTarget.extrinsics.clone();
            isInitialized = true;
        }
        
        SelectionInfo Push(const ImageP image) {
            assert(isInitialized);

            double viewDist = GetAngleOfRotation(image->adjustedExtrinsics, ballTarget.extrinsics);
            
            SelectionInfo info;
            info.closestPoint = ballTarget;
            info.image = image;

            //If we reached a viewpoint...
            if(viewDist < tolerance) {
                SelectionPoint next;
                if(graph.GetNext(ballTarget, next)) {
                    prevTarget = ballTarget;
                    prevDist = 100;
                    ballTarget = next;
                } else {
                    //Ring switch or finish.
                    isFinished = true;
                }
            }

            //Animate ball.
            
            double dist = GetAngleOfRotation(ballPosition, ballTarget.extrinsics);
            double t = cos(dist) * ballSpeed;
            
            Mat newPos(4, 4, CV_64F);
            Lerp(ballPosition, ballTarget.extrinsics, t, newPos);
                
            ballPosition = newPos;
            
            
            error = GetAngleOfRotation(image->adjustedExtrinsics, ballPosition);
            ExtractRotationVector(image->adjustedExtrinsics.inv() * ballPosition, errorVec);
            
            info.dist = viewDist;
            info.isValid = viewDist < tolerance && viewDist < tolerance;
            
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
