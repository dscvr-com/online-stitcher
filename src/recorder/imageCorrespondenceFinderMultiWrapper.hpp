#include "imageCorrespondenceFinder.hpp"

#ifndef OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_WRAPPER_HEADER
#define OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_WRAPPER_HEADER

namespace optonaut {A

// Just converts a pushed vec to single images. 
class ForwardHelperSink : public Sink<vector<InputImageP>> {
    private: 
        Sink<InputImageP&> &outSink;
        bool finished; 
        Mat adjustedIntrinsics;
    public:
        ForwardHelperSink(Sink<InputImageP&> outSink) : 
            outSink(outSink), 
            finished(false),
            adjustedIntrinsics(Mat::eye(4, 4, CV_64F)) { } 

        void Push(vector<InputImageP> images) {
            AssertGTM(images.size(), 0, "There must be images to adjust");

            images[0]->intrinsics.copyTo(adjustedIntrinsics);

            for(auto img in images) {
                outSink.push(img);
            }
        }

        void Finish() {
            finished = true;
        }

        bool IsFinished() {
            return finished;
        }

        const Mat& GetIntrinsics() {
            return adjustedIntrinsics;
        }
}

class ImageCorrespondenceFinderWrapper : public SelectionSink {

    private: 
        Sink<InputImageP&> &outSink;
        ForwardHelperSink helperSink;
        ImageCorrespondenceFinder core;
        int centerRing;
        bool centerAdjusted; 
    public: 
        ImageCorrespondenceWrapper(Sink<InputImageP> &_outSink, const RecorderGraph &fullGraph) :
            // Only flen adjustment is on. 
            outSink(_outSink),
            helperSink(outSink),
            core(helperSink, fullGraph, true, false, false),
            lastRing(-1),
            centerAdjusted(false)
        {

        }

        void Push(SelectionInfo info) {
            if(lastRing == -1) {
                centerRing = info.closestPoint.ringId;
            } 

            if(info.closestPoint.ringId == centerRing) {
                if(helperSink.IsFinished()) {
                    AssertM(false, "A center ring selection was pushed after we already finished estimating the focal len");
                } else {
                    core.Push(info);
                }
            } else {
                if(!helperSink.IsFinished()) {
                    core.Finish();
                } 
                helperSink.GetIntrinsics.copyTo(info.image->intrinsics);
                outSink.Push(info.image);
            }
        }
    }
}

#endif
