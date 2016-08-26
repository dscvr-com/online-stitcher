#include <functional>

using namespace std;

#ifndef OPTONAUT_REDUCE_PROCESSOR_HEADER
#define OPTONAUT_REDUCE_PROCESSOR_HEADER

namespace optonaut {
    /*
     * Provides a streaming reduce process. 
     */
    template <typename InType, typename OutType>
	class ReduceProcessor : Sink<InType> {
	private:
        const function<OutType(const OutType&, const InType&)> reduce;
        OutType state;
    public:
        /*
         * Creates a new instance of this class. 
         *
         * @param op Reduce operation called for each pushed element and the current state.
         * @param initial The initial state. 
         */
        ReduceProcessor(function<OutType(const OutType&, const InType&)> op, 
                const OutType &initial) : 
            reduce(op),
            state(initial) {
        }

        /*
         * Pushes a new element into the reduce processor
         * and updates the state.
         *
         * @param in The element to push.
         *
         * @returns The updated state. 
         */
        const OutType& PushAndGetState(const InType &in) {
            state = reduce(state, in);
            return state;
        }

        virtual void Push(InType in) {
            PushAndGetState(in);
        }

        virtual void Finish() { }

        /*
         * @returns The current state. 
         */
        const OutType& GetState() {
            return state;
        }

    };
}
#endif
