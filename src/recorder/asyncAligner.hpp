#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

#include "../common/image.hpp"
#include "../common/asyncStreamWrapper.hpp"
#include "../math/support.hpp"

#include "aligner.hpp"
#include "recorderGraph.hpp"
#include "sequenceStreamAligner.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_ASYNC_ALIGNMENT_HEADER
#define OPTONAUT_ASYNC_ALIGNMENT_HEADER

namespace optonaut {
	
    class AsyncAlignerCore {    
        private: 
        SequenceStreamAligner core;
        public:
        
        AsyncAlignerCore(RecorderGraph) : core() { }

        Mat operator()(InputImageP in) {
            core.Push(in);
            return core.GetCurrentRotation().clone();
        }
        
        void Postprocess(vector<InputImageP> imgs) const { core.Postprocess(imgs); };
        
        void Finish() { core.Finish(); };
    };

	class AsyncAligner : public Aligner {
	private:
        AsyncAlignerCore core;
        AsyncStream<InputImageP, Mat> worker;

        Mat sensorDiff;
        Mat lastSensor;
        Mat current;

        bool isInitialized;

	public:
		AsyncAligner(RecorderGraph &graph) : core(graph), worker(ref(core)), sensorDiff(Mat::eye(4, 4, CV_64F)), isInitialized(false){ }
       
        bool NeedsImageData() {
            return worker.Finished();
        }

        void Push(InputImageP image) {
            if(!isInitialized) {
                lastSensor = image->originalExtrinsics;
                current = image->originalExtrinsics;
                worker.Push(image);
            }

            if(isInitialized && worker.Finished() && image->IsLoaded()) {
                current = worker.Result() * sensorDiff;
                sensorDiff = Mat::eye(4, 4, CV_64F);
                
                worker.Push(image);
            } else {
                Mat sensorStep = lastSensor.inv() * image->originalExtrinsics;
                sensorDiff = sensorDiff * sensorStep;
                current = current * sensorStep;
                lastSensor = image->originalExtrinsics;
            }
                
            isInitialized = true;
        }

        void Dispose() {
            worker.Dispose();
        }

        Mat GetCurrentRotation() const {
            return current;
        }
        
        void Postprocess(vector<InputImageP> imgs) const { core.Postprocess(imgs); };
        void Finish() { core.Finish(); };
    };
}
#endif
