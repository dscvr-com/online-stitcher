#include <functional>
#include <vector>

#include <algorithm>

#ifndef OPTONAUT_FUNCTIONAL_HEADER
#define OPTONAUT_FUNCTIONAL_HEADER

namespace optonaut {
/*
 * Functional-programming style helpers. 
 */
namespace fun {
    /*
     * Maps an input vector to an output vector, applies a function to each element. 
     *
     * @param input Input vector.
     * @param conv Mapping function, gets applied to each element of the input vector.
     *
     * @returns An output vector containing all mapped elements. 
     */
    template <typename In, typename Out> 
    std::vector<Out> map(const std::vector<In> &input, std::function<Out(const In&)> conv) {
        std::vector<Out> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = conv(input[i]);
        }

        return out;
    }

    /*
     * Flattens a vector of vectors to a single vector. 
     *
     * @param input The vector of vectors to flat.
     *
     * @returns The flattened vector.  
     */
    template <typename In>
    std::vector<In> flat(const std::vector<std::vector<In>> &input) {
        std::vector<In> output;

        for(size_t i = 0; i < input.size(); i++) {
            for(size_t j = 0; j < input[i].size(); j++) {
                output.push_back(input[i][j]);
            }
        }

        return output;
    }

    /*
     * Filters a vector according to a predicate function.
     *
     * @param input The vector to filter.
     * @param a The predicate function, which returns true if the element should be kept in the output. 
     *
     * @returns All elements for which the predicate function returned true. 
     */
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

    /*
     * Reverses the input vector.
     * 
     * @param input The input vector.
     *
     * @returns The reversed vector. 
     */ 
    template <typename In>
    std::vector<In> inverse(const std::vector<const In> &input) {
        std::vector<In> out(input.size());

        for(size_t i = 0; i < input.size(); i++) {
            out[i] = input[input.size() - 1 - i];
        }

        return out;
    }
   
    /*
     * Sorts the given vector according to the key returned by the extractor function.
     *
     * @param input The input vector.
     * @param extractor The key extractor function. 
     *
     * @returns The sorted vector. 
     */ 
    template <typename In, typename Key>
    std::vector<In> orderby(const std::vector<In> &input, std::function<Key(const In&)> extractor) {
        std::vector<In> out = input;

        std::sort(out.begin(), out.end(), [&extractor] (const In &a, const In &b) {
                    return extractor(a) < extractor(b);
                });

        return out;
    }

}}
#endif

