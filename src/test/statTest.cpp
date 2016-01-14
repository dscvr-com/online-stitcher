#include <vector>
#include "../math/stat.hpp"
#include "../common/assert.hpp"

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
    double sumSum = 0;

    for(size_t i = 0; i < in.size(); i++) {
        auto set = in[i];
        double variance = Variance(set); 
        double mean = Mean(set);
        double sum = 0;
        OnlineVariance<double> onlineVariance;

        for(auto d : set) {
            onlineVariance.Push(d);
            sum += d;
            sumSum += d;
        } 

        AssertEQ(variance, expectedVariances[i]);
        AssertEQ(mean, expectedMeans[i]);
        AssertEQ(onlineVariance.Result(), expectedVariances[i]);
        AssertEQ(sum, onlineVariance.Sum());

        pool.Push(variance, set.size(), onlineVariance.Sum());
    }

    AssertEQ(expectedPooledVar, pool.Result());
    AssertEQ(sumSum, pool.Sum());

    cout << "[\u2713] Statistics module." << endl;
}
