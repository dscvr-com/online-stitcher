#include <functional>
#include <deque>
#include "sink.hpp"

using namespace std;

#ifndef OPTONAUT_QUEUE_PROCESSOR_HEADER
#define OPTONAUT_QUEUE_PROCESSOR_HEADER

namespace optonaut {
    /*
     * Queue that calls callbacks as soon as elements
     * enter or leave the queue. 
     */
    template <typename InType>
	class QueueProcessor : public Sink<InType> {
	private:
        const size_t length;

        const function<void(const InType&)> start;
        const function<void(const InType&)> finish;

        deque<InType> buffer;
    public:
        /*
         * Creates a new instance of this class.
         *
         * @param length The length of this queue. 
         * @param onFinish Callback that is called when elements leave the queue. 
         */
        QueueProcessor(size_t length, 
                function<void(const InType&)> onFinish) : 
            length(length),
            start([] (const InType&) {}),
            finish(onFinish) {
        }

        /*
         * Creates a new instance of this class.
         *
         * @param length The length of this queue. 
         * @param onStart Callback that is called when elements enter the queue. 
         * @param onFinish Callback that is called when elements leave the queue. 
         */
        QueueProcessor(size_t length, 
                function<void(const InType&)> onStart,
                function<void(const InType&)> onFinish) : 
            length(length),
            start(onStart),
            finish(onFinish) {
        }
       
        /*
         * Pushes an element into the queue.
         * It will be kept in the queue until it is at the
         * end of the queue. 
         * 
         * @param in The element to push to the queue. 
         */ 
        virtual void Push(InType in) {
            start(in);

            buffer.push_back(in);

            if(buffer.size() > length) {
                finish(buffer.front());
                buffer.pop_front();
            }
        }

        /*
         * Flushes the queue, calls 
         * onFinish for each element, then clears
         * the queue. 
         */
        void Flush() {
            Clear();
        }

        // Alias for flush.
        virtual void Finish() {
            Flush();
        }   

        /*
         * Clears the queue, calls 
         * onFinish for each element, then clears
         * the queue. 
         */
        void Clear() {
            for(auto &b : buffer) {
                 finish(b);
            }

            buffer.clear();
        }
    };
}
#endif
