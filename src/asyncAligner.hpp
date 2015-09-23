#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

#include "image.hpp"
#include "support.hpp"
#include "aligner.hpp"
#include "ringwiseStreamAligner.hpp"
#include <thread>

using namespace cv;
using namespace std;

#ifndef OPTONAUT_ASYNC_ALIGNMENT_HEADER
#define OPTONAUT_ASYNC_ALIGNMENT_HEADER

namespace optonaut {
	class AsyncAligner : public Aligner {
	private:
		RingwiseStreamAligner core;
        ImageP recentImage;
        bool running;
        thread worker;

        Mat sensorDiff;
        Mat lastSensor;
        Mat current;

        mutex m;
        condition_variable sem;

        bool isInitialized;
        bool alignerReady;
        bool dataReady;

        void AlignmentLoop() {
    
            cout << "Async alignment loop running" << endl;
             
            while(running) {
                ImageP recentImage;
                {
                    unique_lock<mutex> lock(m);
                    if(!dataReady) 
                        sem.wait(lock);
                    recentImage = this->recentImage;    
                }

                dataReady = false;

                if(recentImage == NULL)
                    continue;

                core.Push(recentImage);

                {
                    unique_lock<mutex> lock(m);
                    current = core.GetCurrentRotation() * sensorDiff;
                    sensorDiff = Mat::eye(4, 4, CV_64F);
                    //cout << "Update by stream: " << current << endl;
                    alignerReady = true;
                }
            }
        }

	public:
		AsyncAligner(RecorderGraph &graph) : core(graph), running(true), sensorDiff(Mat::eye(4, 4, CV_64F)), isInitialized(false), alignerReady(true) { }
       
        bool NeedsImageData() {
            return alignerReady;
        }

        void Push(ImageP image) {
            if(!isInitialized) {
                lastSensor = image->originalExtrinsics;
                current = image->originalExtrinsics;
                isInitialized = true;
                running = true;
                alignerReady = true;
                worker = thread(&AsyncAligner::AlignmentLoop, this); 
            }
        
            {
                unique_lock<mutex> lock(m);
               
                if(alignerReady) { 
                    recentImage = ImageP(new Image(*image));
                    alignerReady = false;
                   // cout << "Pushing new data to alignment loop " << endl;
                    dataReady = true;
                    sem.notify_one();
                } 
                Mat sensorStep = lastSensor.inv() * image->originalExtrinsics;
                sensorDiff = sensorDiff * sensorStep;
                current = current * sensorStep;
                lastSensor = image->originalExtrinsics;
                //cout << "Update by diff: " << current << endl;
            }
        }

        void Dispose() {
            running = false;
            {
                unique_lock<mutex> lock(m);
                sem.notify_one();
            }
            worker.join();
        }

        Mat GetCurrentRotation() const {
            return current;
        }
        
        void Postprocess(vector<ImageP> imgs) const { core.Postprocess(imgs); };
        void Finish() { core.Finish(); };
    };
}
#endif
