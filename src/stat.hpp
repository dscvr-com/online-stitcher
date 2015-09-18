#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#ifndef OPTONAUT_STAT_HEADER
#define OPTONAUT_STAT_HEADER

namespace optonaut {

    template <typename T>
    static inline T Average(const std::deque<T> &in, float trim = 0) {
        size_t start = in.size() * trim;
        size_t end = in.size() * (1 - trim);
        std::vector<T> buf(in.begin(), in.end());
        std::sort(buf.begin(), buf.end());

        T value = buf[start];

        for(size_t i = start + 1; i < end; i++) {
           value += buf[i]; 
        }

        value /= (end - start);
        
        return value;
    };
    
    template <typename T>
    static inline T Median(const std::deque<T> &in) {
        std::vector<T> buf(in.begin(), in.end());
        std::sort(buf.begin(), buf.end());

        return buf[buf.size() / 2];
    };
}

#endif
