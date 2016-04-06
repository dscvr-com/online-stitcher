/*
 * Statistics module. Contains classes related to statistics. 
 */

#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#include "../common/assert.hpp"

#ifndef OPTONAUT_STAT_HEADER
#define OPTONAUT_STAT_HEADER

namespace optonaut {

    /*
     * Class to calculate the the variance of a data set without having all values available at the same time. 
     * The implemented algorithm is called "Welford's online variance algorithm". Bessel correction is applied. 
     *
     * For basic usage, push values of the dataset into the Push method. Then query the result. 
     */
    template <typename T>
    class OnlineVariance {
        size_t n = 0; 
        T mean = 0;
        T m2 = 0;
        T sum = 0;

    public: 
        void Push(const T &x) {
            n++;
            sum += x;
            T delta = x - mean; 
            mean = mean + delta / n;
            m2 = m2 + delta * (x - mean);
        }

        T Result() const {
            AssertGTM(n, (size_t)0, "Variance is undefined");
            if(n == 1)
                return 0;
            return m2 / (n - 1);
        }

        T Sum() const {
            return sum;
        }
    };

    /*
     * Calculates pooled variance of independent measurements
     * of the same population with different means but assumes same variances.  
     * We remove the bessel correction here. 
     *
     * For use, pass pool variance, sum and pool size to the Push method, then query results. 
     */
    template <typename T>
    class VariancePool {
        struct Measurement {
            T s;
            size_t n;
            T sum;

            Measurement(T s, size_t n, T sum) : s(s), n(n), sum(sum) {}
        };

        std::vector<Measurement> measurements;
    public:

        void Push(T variance, size_t poolSize, T sum) {
            measurements.emplace_back(variance, poolSize, sum);
        }

        size_t Count() const {
            return measurements.size();
        }

        const std::vector<Measurement> &GetMeasurements() const {
            return measurements;
        }

        T Sum() const {
            T sum = 0;

            for(auto m : measurements) {
                sum += m.sum;
            }

            return sum;
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

    /*
     * Calculates the mean of a given dataset. Supports trimming. 
     */
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
  
    /*
     * Calculates bessel corrected variance of a dataset. Supports trimming. 
     */ 
    template <typename T>
    static inline T Variance(const std::deque<T> &in, float trim = 0) {
        size_t start = in.size() * trim;
        size_t end = in.size() * (1 - trim);

        AssertGT(end - start, (size_t)1);

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
    
    /*
     * Calculates the median of a dataset. 
     */
    template <typename T>
    static inline T Median(const std::deque<T> &in) {
        std::vector<T> buf(in.begin(), in.end());
        std::sort(buf.begin(), buf.end());

        return buf[buf.size() / 2];
    };


}

#endif
