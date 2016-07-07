#include <thread>

#include "progressCallback.hpp"

using namespace std;

#ifndef OPTONAUT_RINGWISE_PROCESSOR_HEADER
#define OPTONAUT_RINGWISE_PROCESSOR_HEADER

namespace optonaut {

    /*
     * Helper for processing pairs in a ring data structure. 
     */
    template <typename InType>
	class RingProcessor {
	private:
        const size_t distance;
        const size_t prefixSize;
        size_t prefixCounter;

        const function<void(const InType&)> start;
        const function<void(const InType&, const InType&)> process;
        const function<void(const InType&)> finish;

        deque<InType> buffer;
        deque<InType> prefix;
        
        void PushInternal(const InType &in) {

            if(prefix.size() < prefixSize) {
                prefix.push_back(in);
            }

            buffer.push_back(in);

            if(buffer.size() > distance) {

                process(buffer.front(), buffer.back());
                
                if(prefixCounter == 0) {
                    finish(buffer.front());
                } else {
                    prefixCounter--;
                }

                buffer.pop_front();
            }
        }

    public:

        /*
         * Creates a new instance of this class.
         *
         * @param dist Distance between elements that should be combined (in indices)
         * @param prefix Overlap at the beginning/end of the ring. 
         * @param onStart Callback executed whenever an element is pushed into this ring processor. 
         * @param process Callback executed for each element pair to be combined. 
         * @param onFinish Callback executed whenever all other processing steps for an element are finished and
         *                 the element is removed from the RingProcessor. 
         */
        RingProcessor(size_t dist, size_t prefix,
                function<void(const InType&)> onStart,
                function<void(const InType&, const InType&)> process,
                function<void(const InType&)> onFinish) : 
            distance(dist), 
            prefixSize(prefix),
            prefixCounter(prefix),
            start(onStart),
            process(process),
            finish(onFinish) {
            //Necassary for the below implementation. 
            Assert(prefixSize <= dist);
        }
       
        /*
         * Convenience overload for the above constructor. Uses a minimal prefix and no onStart callback. 
         */ 
        RingProcessor(size_t dist, 
                function<void(const InType&, const InType&)> process,
                function<void(const InType&)> onFinish) : 
            distance(dist), 
            prefixSize(dist),
            prefixCounter(dist),
            start([] (const InType &) {}),
            process(process),
            finish(onFinish) {
            //Necassary for the below implementation. 
            Assert(prefixSize <= dist);
        }

        /*
         * Pushes all elements of a vector into this RingBuffer and flushes the
         * RingBuffer afterwards. 
         *
         * @param in The vector containing the elements to process. 
         * @param prog The progress callback to report progress to (optional). 
         */
        void Process(const vector<InType> &in, ProgressCallback &prog = ProgressCallback::Empty) {
            for(size_t i = 0; i < in.size(); i++) {
                Push(in[i]);
                prog((float)i / (float)in.size());
            }

            Flush();

            prog(1);
        }

        /*
         * Pushes a single element to this ring buffer. 
         *
         * @param in The element to push. 
         */
        void Push(const InType &in) {
            start(in);
            PushInternal(in);
        }

        /*
         * Flushes this ring buffer. In other words, treat the last element
         * pushed as the final element and close the ring by processing all 
         * remeaing pairs.
         *
         * @param push If true, all prefix elements are pushed again (optional).
         */
        void Flush(bool push = true) {
            for(auto &pre : prefix) {
                if(push)
                    PushInternal(pre);
            }

            Clear();
        }

        /*
         * Clears the ring buffer, without processing
         * any further pairs. Finish is still called for each element that is still
         * in the buffer. 
         */
        void Clear() {
            for(auto &b : buffer) {
                 finish(b);
            }

            prefixCounter = prefixSize; 

            prefix.clear();
            buffer.clear();
        }
    };
}
#endif
