#include "../common/sink.hpp"
#include "stereoGenerator.hpp"
#include "imageReselector.hpp"
#include "imageCorrespondenceFinder.hpp"
#include "asyncTolerantRingRecorder.hpp"
#include "imageSelector.hpp"
#include "imageLoader.hpp"
#include "coordinateConverter.hpp"

#ifndef OPTONAUT_RECORDER_2_HEADER
#define OPTONAUT_RECORDER_2_HEADER

namespace optonaut {
class SelectionInfoToImageSink : public MapSink<SelectionInfo, InputImageP> {
    public: 
        SelectionInfoToImageSink(Sink<InputImageP> &outSink) : 
            MapSink<SelectionInfo, InputImageP>([](SelectionInfo in) {
            return in.image;
        }, outSink) { }
};

class Recorder2 {

    private: 
        const RecorderGraphGenerator generator;
        RecorderGraph graph;
        RecorderGraph halfGraph;
        RecorderGraph previewGraph;
        
        std::vector<Mat> allRotations;

        // order of operations, read from bottom to top. 
        // Stitchers for left and right result.
        AsyncRingStitcher leftStitcher;
        AsyncRingStitcher rightStitcher;
        // Creates stereo images
        StereoGenerator stereoGenerator;
        // Re-selects aligned images according to given graph
        ImageReselector reselector;
        // Finds image correspondence and performs exposure adjusting and alignment
        ImageCorrespondenceFinder adjuster;
        // Collects given input images for the center ring and computes a 
        // low-resolution preview image. 
        AsyncTolerantRingRecorder previewStitcher;
        // Converts SelectionInfo struct to InputImageP
        SelectionInfoToImageSink selectionToImageConverter;
        // Forwards the given image to previewStitcher AND correspondenceFinder
        TeeSink<SelectionInfo> previewTee;
        // Selects good images
        FeedbackImageSelector selector;
        // Loads the image from the data ref
        ImageLoader loader;
        // Converts input data to stitcher coord frame
        CoordinateConverter converter;

    public:
        Recorder2(const Mat &base, const Mat &zeroWithoutBase, 
                  const Mat &intrinsics,
                  const int graphConfig = RecorderGraph::ModeAll, 
                  const double tolerance = 1.0) : 
            generator(),
            graph(generator.Generate(
                    intrinsics, 
                    graphConfig, 
                    RecorderGraph::DensityNormal, 
                    0, 8)),
            halfGraph(RecorderGraphGenerator::Sparse(graph, 2)),
            previewGraph(generator.Generate(
                    intrinsics, RecorderGraph::ModeCenter,
                    RecorderGraph::DensityHalf, 0, 8)),
            // TODO - Seperate mapping for all rings with seperate ring stitchers. 
            allRotations(fun::map<SelectionPoint*, Mat>(
               halfGraph.GetTargetsById(), 
               [](const SelectionPoint* x) {
                    return x->extrinsics;
               })), 
            leftStitcher(allRotations, 1200, false),
            rightStitcher(allRotations, 1200, false),
            stereoGenerator(leftStitcher, rightStitcher, halfGraph), 
            reselector(stereoGenerator, halfGraph),
            adjuster(reselector, graph),
            previewStitcher(previewGraph),
            selectionToImageConverter(previewStitcher),
            previewTee(selectionToImageConverter, adjuster),
            selector(graph, previewTee, 
                Vec3d(
                    M_PI / 64 * tolerance, 
                    M_PI / 128 * tolerance, 
                    M_PI / 16 * tolerance)),
            loader(selector), 
            converter(base, zeroWithoutBase, loader)
        { 
            size_t imagesCount = graph.Size();

            AssertEQM(UseSomeMemory(1280, 720, imagesCount), imagesCount, 
                    "Successfully pre-allocate memory");
        } 

        virtual void Push(InputImageP image) {
            Log << "Received Image. ";
            AssertM(!selector.IsFinished(), "Warning: Push after finish - this is probably a racing condition");
            
            converter.Push(image);


        }

        virtual void Finish() {
            converter.Finish();
        }

        StitchingResultP GetPreviewImage() {
            return previewStitcher.Finalize();
        }

        StitchingResultP GetLeftResult() {
            return leftStitcher.Finalize();
        }

        StitchingResultP GetRightResult() {
            return rightStitcher.Finalize();
        }

        bool RecordingIsFinished() {
            return selector.IsFinished();
        }

        const RecorderGraph& GetRecorderGraph() {
            return graph;
        }
        
        Mat GetBallPosition() const {
            return converter.ConvertFromStitcher(selector.GetBallPosition());
        }
        
        SelectionInfo GetCurrentKeyframe() const {
            return selector.GetCurrent();
        }
        
        double GetDistanceToBall() const {
            return selector.GetError();
        }
        
        const Mat &GetAngularDistanceToBall() const {
            return selector.GetErrorVector();
        }
};
}


#endif
