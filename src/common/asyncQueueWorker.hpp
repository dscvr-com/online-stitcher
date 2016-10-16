#include <thread>
#include <condition_variable>
#include "support.hpp"
#include "sink.hpp"

using namespace std;

#ifndef OPTONAUT_ASYNC_QUEUE_HEADER
#define OPTONAUT_ASYNC_QUEUE_HEADER

namespace optonaut {
    /*
     * Worker thread wrapper that asynchronously calls a function for 
     * each element in a queue. 
     *
     * @tparam InType The type of the elements in the processor queue. 
     */
    template <typename InType>
	class AsyncQueue : Sink<InType> {
	private:
        function<void(InType)> core;
        deque<InType> inData;
        bool running;
        bool isInitialized;
        bool cancel;
        thread worker;

        mutex m;
        condition_variable sem;

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
        /*
         * Creates a new instance of this class. The processing
         * thread is started immideately. 
         * 
         * @param core The function to be called for each element 
         * asynchronously. 
         */
		AsyncQueue(function<void(InType)> core) : 
            core(core), 
            inData(), 
            running(false),
            isInitialized(false), 
            cancel(false),
            m(), 
            sem() { }
       
        /*
         * Adds a new element to the processing queue.
         *
         * @param in The element to add.  
         * @returns The size of the queue. 
         */
        int PushAndGetQueueSize(InType in) {
            int size = 0;
            {
                unique_lock<mutex> lock(m);
                inData.push_back(in);
                size = inData.size();
                sem.notify_one();
            }
            
            if(!isInitialized) {
                cancel = false;
                isInitialized = true;
                running = true;
                worker = thread(&AsyncQueue::WorkerLoop, this); 
            }
            
            return size;
        }
      
        virtual void Push(InType in) {
            PushAndGetQueueSize(in);
        }

        /*
         * @returns true if this worker is running, 
         * else false. 
         */ 
        bool IsRunning() {
            return running;
        }
       
        /*
         * Finishes processing of all elements that are currently 
         * in the queue, then exits. 
         */
        virtual void Finish() {
            if(!running)
                return;
            
            running = false;
            {
                unique_lock<mutex> lock(m);
                sem.notify_one();
            }
            worker.join();
        }

        /*
         * Finishes the current operation, then clears the queue
         * and exits. 
         */
        void Dispose() {
            cancel = true;
            Finish();
        }
    };
    
    template <typename DataType> class AsyncSink : public Sink<DataType> {
    private:
        Sink<DataType> &out;
        AsyncQueue<DataType> queue;
        bool interceptFinish;
    public:
        AsyncSink(Sink<DataType> &out, bool interceptFinish = false) :
        out(out),
        queue([&out](const DataType& item){ out.Push(item); }),
        interceptFinish(interceptFinish) { }
          
        virtual void Push(DataType in) {
            queue.Push(in);
        }

        virtual void Finish() {
            queue.Finish();
            if(!interceptFinish) {
                out.Finish();
            }
        }
    };
}
#endif
