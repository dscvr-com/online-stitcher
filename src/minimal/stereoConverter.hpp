#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "../common/functional.hpp"
#include "../recorder/recorderGraph.hpp"
#include "../io/io.hpp"
#include "../stitcher/multiringStitcher.hpp"
#include "../stitcher/stitchingResult.hpp"

using namespace std;
using namespace cv;

#ifndef OPTONAUT_MINIMAL_STEREO_CONVERTER
#define OPTONAUT_MINIMAL_STEREO_CONVERTER

namespace optonaut {
namespace minimal {
class StereoConverter {
    public:
    /*
     * Convenience wrapper for stitch method that writes the output to files directly. 
     */
    static void StitchAndWrite(const vector<InputImageP> &images, 
            const RecorderGraph &graph, const string &name) {
        std::pair<StitchingResultP, StitchingResultP> stereoRes; 
        
        stereoRes = Stitch(images, graph);

        imwrite("dbg/stereo_" + name + "_left.jpg", stereoRes.first->image.data);
        imwrite("dbg/stereo_" + name + "_right.jpg", stereoRes.second->image.data);
    }

    /*
     * Minimal implementation of the recording, stereo and stitching part. 
     *
     * From a list of images and a recorder graph, creates two stereo panoramas. 
     */
    static std::pair<StitchingResultP, StitchingResultP>
    Stitch(const vector<InputImageP> &images, const RecorderGraph &graph) {
    
        MonoStitcher stereoConverter;
        vector<vector<StereoImage>> stereoRings;

        auto ForwardToStereoProcess = 
            [&] (const SelectionInfo &a, const SelectionInfo &b) {

            StereoImage stereo;
            SelectionEdge dummy;
            
            if(!graph.GetEdge(a.closestPoint, b.closestPoint, dummy)) {
                cout << "Skipping pair due to misalignment." << endl;
                return;
            }

            stereoConverter.CreateStereo(a, b, stereo);

            while(stereoRings.size() <= a.closestPoint.ringId) {
                stereoRings.push_back(vector<StereoImage>());
            }
            stereoRings[a.closestPoint.ringId].push_back(stereo);
        };
                
        auto FinishImage = [] (const SelectionInfo) { };

        RingProcessor<SelectionInfo> stereoRingBuffer(1,
                                    ForwardToStereoProcess,
                                    FinishImage);

        BiMap<size_t, uint32_t> imagesToTargets;

        vector<InputImageP> best = graph.SelectBestMatches(images, imagesToTargets); 
        int lastRingId = -1;
        for(auto img : best) {
            SelectionPoint target; 
            //Reassign points
            uint32_t pointId = 0; 
            Assert(imagesToTargets.GetValue(img->id, pointId));
            Assert(graph.GetPointById(pointId, target));
            
            SelectionInfo info;
            info.isValid = true;
            info.closestPoint = target; 
            info.image = img;

            if(lastRingId != -1 && lastRingId != (int)target.ringId) {
                stereoRingBuffer.Flush();
            }

            stereoRingBuffer.Push(info);
            lastRingId = target.ringId;
        }
        stereoRingBuffer.Flush();

        ExposureCompensator dummyCompensator; 
        DummyCheckpointStore leftDummyStore;
        DummyCheckpointStore rightDummyStore;

        MultiRingStitcher leftStitcher(leftDummyStore);
        MultiRingStitcher rightStitcher(rightDummyStore);
        
        vector<vector<InputImageP>> leftRings;
        vector<vector<InputImageP>> rightRings;

        for(vector<StereoImage> ring : stereoRings) {
            leftRings.push_back(fun::map<StereoImage, InputImageP>(ring, [](const StereoImage &x) 
                        {return x.A;}));
            rightRings.push_back(fun::map<StereoImage, InputImageP>(ring, [](const StereoImage &x) 
                        {return x.B;}));
        }

        leftStitcher.InitializeForStitching(leftRings, dummyCompensator);
        rightStitcher.InitializeForStitching(rightRings, dummyCompensator);
        
        auto resLeft = leftStitcher.Stitch(ProgressCallback::Empty); 
        auto resRight = rightStitcher.Stitch(ProgressCallback::Empty); 

        return std::make_pair(resLeft, resRight);
    }
};
}
}
#endif
