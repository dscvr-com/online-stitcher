#include "../io/inputImage.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"

#include "../io/checkpointStore.hpp"

#ifndef OPTONAUT_RING_CLOSER_HEADER
#define OPTONAUT_RING_CLOSER_HEADER

namespace optonaut {
    class RingCloser {

        public: 

        /*
         * Closes a ring by comparing the first and last image and applying the resulting rotation
         * offset to all images. The applied offset is interpolated depending on image position. 
         */
        static inline bool CloseRing(std::vector<InputImageP> ring) {
            PairwiseCorrelator corr;
            
            if(CheckpointStore::DebugStore != nullptr) {
                CheckpointStore::DebugStore->SaveRectifiedImage(ring.front());
                CheckpointStore::DebugStore->SaveRectifiedImage(ring.back());
            }

            auto result = corr.Match(ring.front(), ring.back(), 4, 4, true); 

            // Todo - if the result is invalid or too negative, it might be better to 
            // correlate additional images instead only the last image. 
            if(!result.valid) {
                cout << "Ring closure: Rejected." << endl;
                return false;
            }

            // Todo - this is a quick hack to reject too far away images
            // that would lead to black stripes. A better solution would be to
            // just record more ring. 
            if(result.angularOffset.y > 0.12) {
                cout << "Ring closure: Rejected because it would lead to black stripes"<< endl;
                return false;
            }

            cout << "Ring closure: Adjusting by: " << result.angularOffset.y << endl;

            size_t n = ring.size();

            // Move images according to their position. 
            for(size_t i = 0; i < n; i++) {
                double ydiff = result.angularOffset.y * 
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
