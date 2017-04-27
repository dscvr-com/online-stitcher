#include "imageCorrespondenceFinder.hpp"

#ifndef OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_WRAPPER_HEADER
#define OPTONAUT_IMAGE_CORRESPONDENCE_FINDER_WRAPPER_HEADER

namespace optonaut {

class TrivialSelector : public ImageSink {
    private:
        SelectionSink& outSink;
        const RecorderGraph& graph;
    public:
        TrivialSelector(SelectionSink& _outSink, const RecorderGraph &_graph) :
            outSink(_outSink), graph(_graph) { }

    void Push(InputImageP img) {
        SelectionInfo info;
        info.image = img;
        graph.FindClosestPoint(img->adjustedExtrinsics, info.closestPoint); 

        outSink.Push(info);
    }

    void Finish() {
        outSink.Finish();
    }
};

// Just converts a pushed vec to single images. 
class ForwardHelperSink : public Sink<vector<InputImageP>> {
    private: 
        ImageSink& outSink;
        bool finished; 
        Mat adjustedIntrinsics;
    public:
        ForwardHelperSink(ImageSink& _outSink) : 
            outSink(_outSink), 
            finished(false),
            adjustedIntrinsics(Mat::eye(4, 4, CV_64F)) { } 

        void Push(vector<InputImageP> images) {
            AssertGTM(images.size(), (size_t)0, "There must be images to adjust");

            Log << "Forwarding all";

            images[0]->intrinsics.copyTo(adjustedIntrinsics);

            for(auto img : images) {
                outSink.Push(img);
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
};

class ImageCorrespondenceFinderWrapper : public SelectionSink {

    private: 
        ImageSink& outSink;
        ForwardHelperSink helperSink;
        ImageCorrespondenceFinder core;
        int centerRing;
    public: 
        ImageCorrespondenceFinderWrapper(ImageSink& _outSink, const RecorderGraph& fullGraph) :
            outSink(_outSink),
            helperSink(outSink),
            // Only flen adjustment is on. 
            core(helperSink, fullGraph, true, false, false),
            centerRing(-1)
        {

        }

        void Push(SelectionInfo info) {
            if(centerRing == -1) {
                centerRing = info.closestPoint.ringId;
            } 

            if((int)info.closestPoint.ringId == centerRing) {
                Log << "Forwarding to core";
                if(helperSink.IsFinished()) {
                    AssertM(false, "A center ring selection was pushed after we already finished estimating the focal len");
                } else {
                    core.Push(info);
                }
            } else {
                Log << "Forwarding direct";
                if(!helperSink.IsFinished()) {
                    Log << "Finishing core";
                    core.Finish();
                } 
                helperSink.GetIntrinsics().copyTo(info.image->intrinsics);
                outSink.Push(info.image);
            }
        }
        
        void Finish() {
            Log << "Finishing";
            if(!helperSink.IsFinished()) {
                Log << "Finishing core";
                core.Finish();
            } 

            outSink.Finish();
        }
};
}

#endif
