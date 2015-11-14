#include <functional>
#include <deque>

using namespace std;

#ifndef OPTONAUT_QUEUE_PROCESSOR_HEADER
#define OPTONAUT_QUEUE_PROCESSOR_HEADER

namespace optonaut {
    template <typename InType>
	class QueueProcessor {
	private:
        const size_t length;

        function<void(InType&)> start;
        function<void(InType&)> finish;

        deque<InType> buffer;

    public:

        QueueProcessor(size_t length, 
                function<void(InType&)> onStart,
                function<void(InType&)> onFinish) : 
            length(length),
            start(onStart),
            finish(onFinish) {
        }
        
        void Push(InType &in) {

            start(in);

            buffer.push_back(in);

            if(buffer.size() > length) {
                finish(buffer.front());
                buffer.pop_front();
            }
        }

        void Flush() {
            Clear();
        }

        void Clear() {
            for(auto &b : buffer) {
                 finish(b);
            }

            buffer.clear();
        }
    };
}
#endif
