#include <thread>
#include <condition_variable>
#include "support.hpp"

using namespace std;

#ifndef OPTONAUT_ASYNC_QUEUE_HEADER
#define OPTONAUT_ASYNC_QUEUE_HEADER

namespace optonaut {
    template <typename InType>
	class AsyncQueue {
	private:
        function<void(InType)> core;
        deque<InType> inData;
        bool running;
        bool cancel;
        thread worker;

        mutex m;
        condition_variable sem;

        bool isInitialized;

        void WorkerLoop() {
    
            while(!cancel && (running || inData.size() != 0)) {
                InType elem;
                {
                    unique_lock<mutex> lock(m);
                    while(!cancel && inData.size() == 0 && running)
                        sem.wait(lock);
                    
                    if(cancel || (!running && inData.size() == 0))
                        break;
                    
                    elem = inData.front();
                    inData.pop_front();
                }

                core(elem);
            }
        }

	public:
		AsyncQueue(function<void(InType)> core) : core(core), running(false), isInitialized(false) { }
       
        void Push(InType in) {
            if(!isInitialized) {
                isInitialized = true;
                running = true;
                worker = thread(&AsyncQueue::WorkerLoop, this); 
            }
        
            {
                unique_lock<mutex> lock(m);
                inData.push_back(in);
                sem.notify_one();
            }
        }
        
        bool IsRunning() {
            return running;
        }
        
        // Finishes current queue, then exits.
        void Finish() {
            if(!running)
                return;
            
            running = false;
            {
                unique_lock<mutex> lock(m);
                sem.notify_one();
            }
            worker.join();
        }

        // Finishes current operation, then exit
        void Dispose() {
            cancel = true;
            Finish();
        }
    };
}
#endif
