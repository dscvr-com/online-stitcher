#include "../common/sink.hpp"
#include "../common/imageQueueProcessor.hpp"
#include "../common/asyncQueueWorker.hpp"
#include "recorderGraphGenerator.hpp"
#include "stereoGenerator.hpp"
#include "imageReselector.hpp"
#include "imageCorrespondenceFinder.hpp"
#include "asyncTolerantRingRecorder.hpp"
#include "imageSelector.hpp"
#include "imageLoader.hpp"
#include "recorder2.hpp"
#include "coordinateConverter.hpp"
#include "debugSink.hpp"
#include "storageImageSink.hpp"

#ifndef OPTONAUT_MOTOR_CONTROL_RECORDER_HEADER
#define OPTONAUT_MOTOR_CONTROL_RECORDER_HEADER

namespace optonaut {



class MotorControlRecorder {

    private:
        const Mat zeroWithoutBase;
        const Mat base;
        const Mat intrinsics;

        const RecorderGraphGenerator generator;
        RecorderGraph graph;
        //RecorderGraph halfGraph;
        RecorderGraph previewGraph;

        std::vector<Mat> allRotations;

        // order of operations, read from bottom to top.
        // Stitchers for left and right result.
        //AsyncRingStitcher leftStitcher;
       // AsyncRingStitcher rightStitcher;
        ExposureCompensator exposure;

        StorageImageSink &sink;
        //ImageForPostProcess postProcess;
        ImageQueueProcessor<SelectionInfo> imageSave;
        AsyncSink<SelectionInfo> postProcessImage;
        // Collects given input images for the center ring and computes a
        // low-resolution preview image.
        AsyncTolerantRingRecorder previewStitcher;
        // Converts SelectionInfo struct to InputImageP
        SelectionInfoToImageSink selectionToImageConverter;
        // Forwards the given image to previewStitcher AND correspondenceFinder
        TeeSink<SelectionInfo> previewTee;
        // Writes debug images, if necassary.
        DebugSink debugger;
        // Decouples slow correspondence finiding process from UI
        AsyncSink<SelectionInfo> decoupler;
        // Selects good images
        FeedbackImageSelector selector;
        // Loads the image from the data ref
        ImageLoader loader;
        // Converts input data to stitcher coord frame
        CoordinateConverter converter;

    public:
        MotorControlRecorder(const Mat &_base, const Mat &_zeroWithoutBase,
                  const Mat &_intrinsics,
                  StorageImageSink &sink,
                  const int graphConfig = RecorderGraph::ModeAll,
                  const double tolerance = 1.0,
                  const std::string debugPath = "") :
            zeroWithoutBase(_zeroWithoutBase),
            base(_base),
            intrinsics(_intrinsics),
            generator(),
            graph(generator.Generate(
                    intrinsics,
                    graphConfig,
                    RecorderGraph::DensityNormal,
                    //RecorderGraph::DensityHalf,
                    0, 8)),
           // halfGraph(RecorderGraphGenerator::Sparse(graph, 2)),
            previewGraph(generator.Generate(
                    intrinsics, RecorderGraph::ModeCenter,
                    RecorderGraph::DensityHalf, 0, 8)),
            // TODO - Seperate mapping for all rings with seperate ring stitchers.
           // allRotations(fun::map<SelectionPoint*, Mat>(
           //    halfGraph.GetTargetsById(),
           //    [](const SelectionPoint* x) {
           //         return x->extrinsics;
           //    })),
           // leftStitcher(allRotations, 1200, false),
           // rightStitcher(allRotations, 1200, false),
            sink(sink),
            imageSave(sink),
            postProcessImage(imageSave, false),
            previewStitcher(previewGraph),
            selectionToImageConverter(previewStitcher),
            previewTee(selectionToImageConverter, postProcessImage),
            debugger(debugPath, debugPath.size() == 0, previewTee),
            decoupler(debugger, true),
            selector(graph, decoupler,
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

        // This has to be called after GetPreviewImage.
        virtual void Finish() {
            AssertM(previewStitcher.GetResult() != nullptr, "GetPreviewImage must be called before calling Finish");
            //adjuster.Finish();
            if (imageSave.HasResults()) {
               Log << "Image Save has results";

               Log << "imageSave.postImages size" << sizeof(imageSave.postImages);
                vector<vector<InputImageP>> postRings =
                    graph.SplitIntoRings(imageSave.postImages);

                Log << "postRings " << postRings.size();
                Log << "postRings "  << sizeof(postRings);
                sink.Finish(postRings,  exposure.GetGains());

            }
           postProcessImage.Finish();
        }

        double getTopThetaValue () {
          if (graph.vCenterList.size() == 3 ) {
            return graph.vCenterList[2];
          }
          return 0;
        }
        double getCenterThetaValue () {
          if (graph.vCenterList.size() == 3) {
            return graph.vCenterList[1];
          }
          return 0;
        }
        double getBotThetaValue () {
          if (graph.vCenterList.size() == 3) {
            return graph.vCenterList[0];
          }
          return 0;
        }




        void Cancel() {
            Log << "Cancel, calling converter finish.";
            converter.Finish();
            Log << "Cancel, calling preview finish.";
            previewStitcher.Finish();
            Log << "Cancel, calling adjuster finish.";
            postProcessImage.Finish();
        }

        StitchingResultP GetPreviewImage() {
            // Calling finish here circumvents the chaining.
            // It is safe because our AsyncSink decoupler intercepts finish.
            Log << "Get Preview Image. ";
            converter.Finish();
            Log << "converter finish. ";
            previewStitcher.Finish();
            Log << "preview stitcher finish. ";
            return previewStitcher.Finalize();
        }

/*
        StitchingResultP GetLeftResult() {
            Log << "GetLeftResult" ;
            return leftStitcher.Finalize();
        }

        StitchingResultP GetRightResult() {
            Log << "GetRightResult";
            return rightStitcher.Finalize();
        }

*/
        bool RecordingIsFinished() {
            return selector.IsFinished();
        }

        const RecorderGraph& GetRecorderGraph() {
            return graph;
        }

        // TODO - rather expose selector
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

        bool IsIdle() {
            return selector.IsIdle();
        }

        bool HasStarted() {
            return selector.HasStarted();
        }

        bool IsFinished() {
            return selector.IsFinished();
        }

        void SetIdle(bool isIdle) {
            selector.SetIdle(isIdle);
        }

        uint32_t GetImagesToRecordCount() {
            return (uint32_t)selector.GetImagesToRecordCount();
        }

        uint32_t GetRecordedImagesCount() {
            return (uint32_t)selector.GetRecordedImagesCount();
        }

        // TODO - refactor out
        bool AreAdjacent(SelectionPoint a, SelectionPoint b) {
            SelectionEdge dummy;
            //return halfGraph.GetEdge(a, b, dummy);
            return graph.GetEdge(a, b, dummy);
        }
        vector<SelectionPoint> GetSelectionPoints() const {
            vector<SelectionPoint> converted;
            for(auto ring : graph.GetRings()) {
                ring.push_back(ring.front());
                for(auto point : ring) {
                    SelectionPoint n;
                    n.globalId = point.globalId;
                    n.ringId = point.ringId;
                    n.localId = point.localId;
                    n.extrinsics = converter.ConvertFromStitcher(point.extrinsics);

                    converted.push_back(n);
                }

            }
            return converted;
        }
};
}


#endif
