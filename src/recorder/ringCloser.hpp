#include "../io/inputImage.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"

#include "checkpointStore.hpp"

#ifndef OPTONAUT_RING_CLOSER_HEADER
#define OPTONAUT_RING_CLOSER_HEADER

namespace optonaut {
    class RingCloser {

        public: 

        static inline bool CloseRing(std::vector<InputImageP> ring) {
            PairwiseCorrelator corr;
            
            if(CheckpointStore::DebugStore != nullptr) {
                CheckpointStore::DebugStore->SaveRectifiedImage(ring.front());
                CheckpointStore::DebugStore->SaveRectifiedImage(ring.back());
            }

            auto result = corr.Match(ring.front(), ring.back(), 4, 4, true); 

            if(!result.valid) {
                cout << "Ring closure: Rejected." << endl;
                return false;
            }

            if(result.angularOffset.x < -0.18) {
                cout << "Ring closure: Rejected because it would lead to black stripes"<< endl;
                return false;
            }

            cout << "Ring closure: Adjusting by: " << result.angularOffset.x << endl;

            size_t n = ring.size();

            for(size_t i = 0; i < n; i++) {
                double ydiff = result.angularOffset.x * 
                    (1.0 - ((double)i) / ((double)n));
                Mat correction;
                CreateRotationY(ydiff, correction);
                ring[i]->adjustedExtrinsics = correction * 
                    ring[i]->adjustedExtrinsics;
            }
            return true;
        }
    };
}

#endif
