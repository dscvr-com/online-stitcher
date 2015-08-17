#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "streamAligner.hpp"
#include <thread>

using namespace cv;
using namespace std;

#ifndef OPTONAUT_ASYNC_ALIGNMENT_HEADER
#define OPTONAUT_ASYNC_ALIGNMENT_HEADER

namespace optonaut {
	class AsyncAligner {
	private:
		StreamAligner core;
        ImageP recentImage;
        bool running;
        thread worker;

        Mat sensorDiff;
        Mat lastSensor;
        Mat current;

        mutex m;
        condition_variable dataReady;

        bool isInitialized;

        void AlignmentLoop() {
    
            while(running) {
                ImageP recentImage;
                {
                    unique_lock<mutex> lock(m);
                    dataReady.wait(lock);
                    recentImage = this->recentImage;    
                }

                if(recentImage == NULL)
                    continue;

                core.Push(recentImage);

                {
                    unique_lock<mutex> lock(m);
                    current = core.GetCurrentRotation() * sensorDiff;
                    sensorDiff = Mat::eye(4, 4, CV_64F);
                    cout << "Update by stream: " << current << endl;
                }
            }
        }

	public:
		AsyncAligner() : core(), running(true), sensorDiff(Mat::eye(4, 4, CV_64F)), isInitialized(false) { }
        
        void Push(ImageP image) {
            if(!isInitialized) {
                lastSensor = image->extrinsics;
                current = image->extrinsics;
                isInitialized = true;
                worker = thread(&AsyncAligner::AlignmentLoop, this); 
            }
        
            {
                unique_lock<mutex> lock(m);
                
                recentImage = ImageP(new Image);
                recentImage->id = image->id;
                recentImage->img = image->img;
                recentImage->extrinsics = image->extrinsics.clone();
                recentImage->intrinsics = image->intrinsics;
                recentImage->source = image->source;
                
                dataReady.notify_one(); 
                sensorDiff = sensorDiff * lastSensor.t() * image->extrinsics; 
                current = current * sensorDiff;
                lastSensor = image->extrinsics;
                cout << "Update by diff: " << current << endl;
            }
        }

        void Dispose() {
            running = false;
            {
                unique_lock<mutex> lock(m);
                dataReady.notify_one();
            }
            worker.join();
        }

        Mat GetCurrentRotation() const {
            return current;
        }
    };
}
#endif
