#ifndef OPTONAUT_RECORDER_PARAM_INFO_HEADER
#define OPTONAUT_RECORDER_PARAM_INFO_HEADER

namespace optonaut {
struct RecorderParamInfo {
    const double graphHOverlap;
    const double graphVOverlap;
    const double stereoHBuffer;
    const double stereoVBuffer;
    const double tolerance;
    const bool halfGraph;
    
    RecorderParamInfo() :
        graphHOverlap(0.7),
        graphVOverlap(0.25),
        stereoHBuffer(0.6),
        stereoVBuffer(-0.05),
        tolerance(2.0),
        halfGraph(true) { }

    RecorderParamInfo(const double graphHOverlap, const double graphVOverlap, const double stereoHBuffer, const double stereoVBuffer, const double tolerance, const bool halfGraph)  :
        graphHOverlap(graphHOverlap),
        graphVOverlap(graphVOverlap),
        stereoHBuffer(stereoHBuffer),
        stereoVBuffer(stereoVBuffer),
        tolerance(tolerance),
        halfGraph(halfGraph) { }
};
}

#endif
