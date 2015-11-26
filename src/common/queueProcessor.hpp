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

        const function<void(const InType&)> start;
        const function<void(const InType&)> finish;

        deque<InType> buffer;
    public:
        QueueProcessor(size_t length, 
                function<void(const InType&)> onFinish) : 
            length(length),
            start([] (const InType&) {}),
            finish(onFinish) {
        }

        QueueProcessor(size_t length, 
                function<void(const InType&)> onStart,
                function<void(const InType&)> onFinish) : 
            length(length),
            start(onStart),
            finish(onFinish) {
        }
        
        void Push(const InType &in) {

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
