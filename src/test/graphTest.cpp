#include <vector>
#include <string>
#include "../recorder/recorder.hpp"
#include "../common/intrinsics.hpp"

using namespace std;
using namespace optonaut;

int main(int, char**) {

    RecorderGraphGenerator generator; 
    RecorderGraph fullGraph = generator.Generate(
            iPhone5Intrinsics, 
            RecorderGraph::ModeAll, 
            RecorderGraph::DensityNormal,
            0, 2);

    RecorderGraph fullGraph_2 = RecorderGraphGenerator::Sparse(fullGraph, 1); 

    auto targets1 = fullGraph.GetTargetsById();
    auto targets2 = fullGraph_2.GetTargetsById();

    // Test that sparse without skip produces same
    // result (basic invariance).
    AssertEQM(targets1.size(), 
              targets2.size(), 
              "Same count of elements");

    for(size_t i = 0; i < targets1.size(); i++) {
        AssertEQ(targets1[i]->globalId, targets2[i]->globalId);
        AssertEQ(targets1[i]->localId, targets2[i]->localId);
        AssertEQ(targets1[i]->ringId, targets2[i]->ringId);
        AssertEQ(targets1[i]->ringSize, targets2[i]->ringSize);
        AssertEQ(targets1[i]->hPos, targets2[i]->hPos);
        AssertEQ(targets1[i]->vPos, targets2[i]->vPos);
        AssertEQ(targets1[i]->hFov, targets2[i]->hFov);
        AssertEQ(targets1[i]->vFov, targets2[i]->vFov);
        //AssertEQ(targets1[i]->extrinsics, targets2[i]->extrinsics);
    }
   
    // Test half graph.  
    RecorderGraph halfGraph = RecorderGraphGenerator::Sparse(fullGraph, 2); 
    auto targetsH = halfGraph.GetTargetsById();

    AssertEQM(targets2.size(), 
              targetsH.size() * 2, 
              "Same count of elements");

    for(size_t j = 0, i = 0; j < targetsH.size(); j++, i+=2) {
        AssertEQ(targetsH[j]->globalId, targets2[i]->globalId / 2);
        AssertEQ(targetsH[j]->localId, targets2[i]->localId / 2);
        AssertEQ(targetsH[j]->ringId, targets2[i]->ringId);
        AssertEQ(targetsH[j]->ringSize, targets2[i]->ringSize / 2);
        AssertEQ(targetsH[j]->hPos, targets2[i]->hPos);
        AssertEQ(targetsH[j]->vPos, targets2[i]->vPos);
        AssertEQ(targetsH[j]->hFov, targets2[i]->hFov * 2);
        AssertEQ(targetsH[j]->vFov, targets2[i]->vFov);
        //AssertEQ(targets1[i]->extrinsics, targets2[i]->extrinsics);
    }
    
    
    cout << "[\u2713] Recorder graph toolkit." << endl;
}
