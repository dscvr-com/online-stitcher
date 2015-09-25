#include <thread>
#include "support.hpp"

using namespace std;

#ifndef OPTONAUT_ASYNC_STREAM_HEADER
#define OPTONAUT_ASYNC_STREAM_HEADER

namespace optonaut {
    template <typename InType, typename OutType>
	class AsyncStream {
	private:
        function<OutType(InType)> core;
        OutType outData;
        InType inData;
        bool running;
        thread worker;

        mutex m;
        condition_variable sem;

        bool isInitialized;
        bool workerReady;
        bool dataReady;

        void WorkerLoop() {
    
            while(running) {
                InType inData;
                {
                    unique_lock<mutex> lock(m);
                    if(!dataReady) 
                        sem.wait(lock);
                    inData = this->inData;
                    dataReady = false;
                }

                if(!running)
                    break;

                OutType outData = core(inData); 

                {
                    unique_lock<mutex> lock(m);
                    this->outData = outData;
                    workerReady = true;
                }
            }
        }

	public:
		AsyncStream(function<OutType(InType)> core) : core(core), running(false), isInitialized(false), workerReady(true) { }
       
        void Push(InType in) {
            if(!isInitialized) {
                isInitialized = true;
                running = true;
                workerReady = true;
                worker = thread(&AsyncStream::WorkerLoop, this); 
            }
        
            {
                unique_lock<mutex> lock(m);
               
                if(workerReady) { 
                    inData = in;
                    workerReady = false;

                    dataReady = true;
                    sem.notify_one();
                } 
            }
        }

        bool Finished() {
            return workerReady;
        }

        OutType Result() {
            return outData;
        }

        void Dispose() {
            if(!running)
                return;

            running = false;
            {
                unique_lock<mutex> lock(m);
                sem.notify_one();
            }
            worker.join();
        }
    };
}
#endif
