#include <functional>
#include <vector>

#include <algorithm>

#ifndef OPTONAUT_FUNCTIONAL_HEADER
#define OPTONAUT_FUNCTIONAL_HEADER

namespace optonaut {
namespace fun {
    template <typename In, typename Out> 
    std::vector<Out> map(const std::vector<In> &input, std::function<Out(const In&)> conv) {
        std::vector<Out> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = conv(input[i]);
        }

        return out;
    }

    template <typename In> 
    std::vector<In> filter(const std::vector<In> &input, std::function<bool(const In&)> a) {
        std::vector<In> out;

        for(size_t i = 0; i < input.size(); i++) {
            if(a(input[i])) {
                out.push_back(input[i]);
            }
        }

        return out;
    }

    template <typename In>
    std::vector<In> inverse(const std::vector<const In> &input) {
        std::vector<In> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = input[input.size() - 1 - i];
        }

        return out;
    }
    
    template <typename In, typename Key>
    std::vector<In> orderby(const std::vector<In> &input, std::function<Key(const In&)> extractor) {
        std::vector<In> out = input;

        std::sort(out.begin(), out.end(), [&extractor] (In &a, In &b) {
                    return extractor(a) < extractor(b);
                });

        return out;
    }

}}
#endif

