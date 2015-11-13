#include <vector>
#include "math/stat.hpp"
#include "common/assert.hpp"

using namespace std;
using namespace optonaut;

int main(int, char**) {

    vector<deque<double>> in = {{31, 30, 29}, 
                                 {42, 41, 40, 39}, 
                                 {31, 28},
                                 {23, 22, 21, 19, 18}, 
                                 {21, 20, 19, 18,17}};

    vector<double> expectedMeans = {30, 40.5, 29.5, 20.6, 19};
    vector<double> expectedVariances = {1, 1.66667, 4.5, 4.3, 2.5};
    double expectedPooledVar = 2.76429;
    VariancePool<double> pool;

    for(size_t i = 0; i < in.size(); i++) {
        auto set = in[i];
        double variance = Variance(set); 
        double mean = Mean(set);
        OnlineVariance<double> onlineVariance;

        for(auto d : set) {
            onlineVariance.Push(d);
        } 

        AssertEQ(variance, expectedVariances[i]);
        AssertEQ(mean, expectedMeans[i]);
        AssertEQ(onlineVariance.Result(), expectedVariances[i]);

        pool.Push(variance, set.size());
    }

    AssertEQ(expectedPooledVar, pool.Result());

    cout << "[\u2713] Statistic module." << endl;

}
