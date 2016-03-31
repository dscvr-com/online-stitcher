#include <thread>
#include "support.hpp"

using namespace std;

#ifndef OPTONAUT_ASYNC_STREAM_HEADER
#define OPTONAUT_ASYNC_STREAM_HEADER

namespace optonaut {

    /*
     * Processes elements asynchronously, but without a queue. 
     * If elements are pushed faster than they can be processed, they are discarded. 
     *
     * @tparam InType The input type. 
     * @tparam OutType the output type. 
     */
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
        /*
         * Creates a new instance of this class. 
         *
         * @param core The processing function to call for each element. 
         */
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

        /*
         * @returns True if the worker is idle. 
         */
        bool Finished() {
            return workerReady;
        }

        /*
         * @returns The result of the last processing run. 
         */
        OutType Result() {
            return outData;
        }
       
        /*
         * True, if the worker thread is active. 
         */ 
        bool IsRunning() {
            return running;
        }

        /*
         * Waits for execution of the current task,
         * then terminates the worker thread. 
         */
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
