#include <functional>

using namespace std;

#ifndef OPTONAUT_REDUCE_PROCESSOR_HEADER
#define OPTONAUT_REDUCE_PROCESSOR_HEADER

namespace optonaut {
    template <typename InType, typename OutType>
	class ReduceProcessor {
	private:
        function<OutType(OutType&, InType&)> reduce;
        OutType state;
    public:
        ReduceProcessor(function<OutType(OutType&, InType&)> op, 
                const OutType &initial) : 
            reduce(op),
            state(initial) {
        }

        const OutType& Push(InType &in) {
            state = reduce(state, in);
            return state;
        }

        const OutType& GetState() {
            return state;
        }

    };
}
#endif
