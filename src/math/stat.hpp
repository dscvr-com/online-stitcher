#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#include "../common/assert.hpp"

#ifndef OPTONAUT_STAT_HEADER
#define OPTONAUT_STAT_HEADER

namespace optonaut {

    //Welford's sweet online variance algorithm. Approved by Grandmaster Knuth. 
    //This algorithm is bessel-corrected. 
    template <typename T>
    class OnlineVariance {
        size_t n = 0; 
        T mean = 0;
        T m2 = 0;

    public: 
        void Push(T x) {
            n++;
            T delta = x - mean; 
            mean = mean + delta / n;
            m2 = m2 + delta * (x - mean);
        }

        T Result() {
            AssertGTM(n, 0, "Variance is undefined");
            if(n == 1)
                return 0;
            return m2 / (n - 1);
        }
    };

    //Calculate pooled variance of independent measurements
    //of the same population with different means but assumes same variances.  
    //We remove the bessel correction here. 
    template <typename T>
    class VariancePool {
        struct Measurement {
            T s;
            size_t n;

            Measurement(T s, size_t n) : s(s), n(n) {}
        };

        std::vector<Measurement> measurements;
    public:

        void Push(T variance, size_t poolSize) {
            measurements.emplace_back(variance, poolSize);
        }

        size_t Count() const {
            return measurements.size();
        }

        T Result() const {
            T nom = 0;
            T den = -((int)measurements.size());

            for(auto m : measurements) {
                nom += (m.n - 1) * m.s;
                den += m.n;
            } 

            return nom / den; 
        }
    };

    template <typename T>
    static inline T Mean(const std::deque<T> &in, float trim = 0) {
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
   
    //Bessel-Corrected variance.  
    template <typename T>
    static inline T Variance(const std::deque<T> &in, float trim = 0) {
        size_t start = in.size() * trim;
        size_t end = in.size() * (1 - trim);

        AssertGT(end - start - 1, 1);

        T average = Mean(in, trim);
        
        T x = in[start] - average;
        T value = x * x;

        for(size_t i = start + 1; i < end; i++) {
            x = in[i] - average;
            value += x * x;
        }

        value /= (end - start - 1);
        
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
