#include <functional>
#include <vector>

#include <algorithm>

#ifndef OPTONAUT_FUNCTIONAL_HEADER
#define OPTONAUT_FUNCTIONAL_HEADER

namespace optonaut {
namespace fun {
    template <typename In, typename Out> 
    vector<Out> map(const vector<In> &input, std::function<Out(const In&)> conv) {
        vector<Out> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = conv(input[i]);
        }

        return out;
    }

    template <typename In>
    vector<In> inverse(const vector<const In> &input) {
        vector<In> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = input[input.size() - 1 - i];
        }

        return out;
    }
    
    template <typename In, typename Key>
    vector<In> orderby(const vector<In> &input, std::function<Key(const In&)> extractor) {
        vector<In> out = input;

        std::sort(out.begin(), out.end(), [&extractor] (In &a, In &b) {
                    return extractor(a) < extractor(b);
                });

        return out;
    }

}}
#endif

