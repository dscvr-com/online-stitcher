#include <thread>

using namespace std;

#ifndef OPTONAUT_RINGWISE_PROCESSOR_HEADER
#define OPTONAUT_RINGWISE_PROCESSOR_HEADER

namespace optonaut {
    template <typename InType>
	class RingProcessor {
	private:
        size_t distance;
        size_t prefixSize;

        function<void(InType&, InType&)> process;
        function<void(InType&)> finish;

        deque<InType> buffer;
        deque<InType> prefix;

    public:

        RingProcessor(size_t dist, 
                function<void(InType&, InType&)> process,
                function<void(InType&)> onFinish) : 
            distance(dist), 
            prefixSize(dist),
            process(process),
            finish(onFinish) {
            //Necassary for the below implementation. 
            assert(prefixSize <= dist);
        }

        void Push(InType &in) {

            if(prefix.size() < prefixSize) {
                prefix.push_back(in);
            }

            buffer.push_back(in);

            if(buffer.size() > distance) {

                process(buffer.front(), buffer.back());
                
                if(prefixSize == 0) {
                    finish(buffer.front());
                } else {
                    prefixSize--;
                }

                buffer.pop_front();
            }
        }

        void Flush() {
            for(auto &pre : prefix) {
                Push(pre);
                finish(pre);
            } 

            prefixSize = distance; 

            prefix.clear();
            buffer.clear();
        }
    };
}
#endif
