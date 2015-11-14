#include <vector>
#include <string>
#include "../common/queueProcessor.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/assert.hpp"

using namespace std;
using namespace optonaut;

int main(int, char**) {
    //Ring processor checks. 

    vector<int> input = {0, 1, 2, 3, 4};
    vector<int> ones = {1, 1, 1, 1, 1};
    vector<vector<int>> adj = {{0, 1, 0, 0, 0},
                               {0, 0, 1, 0, 0},
                               {0, 0, 0, 1, 0},
                               {0, 0, 0, 0, 1},
                               {1, 0, 0, 0, 0}};
    vector<int> check1, check2;
    vector<vector<int>> checkAdj;


    auto check = [](vector<int> &in, string message) {
        for(size_t i = 0; i < in.size(); i++) {
            AssertEQM(in[i], 0, message);
        }
    };

    auto decrease1 = [&](int in) {
        check1[in]--;
    };
    
    auto decrease2 = [&](int in) {
        check2[in]--;
    };
    
    auto decreaseAdj = [&](int a, int b) {
        checkAdj[a][b]--;
    };

    auto setupTest = [&]() {
        check1 = ones;
        check2 = ones;
        checkAdj = adj;
    };

    auto checkTest = [&]() {
        check(check1, "All elements are registered on start.");
        check(check2, "All elements are registered on end.");
        for(auto row : checkAdj) {
            check(row, "All elements are pairwise combined.");
        }
    };

    RingProcessor<int> ring(1, 1, decrease1, decreaseAdj, decrease2);

    setupTest();
    ring.Process(input);
    ring.Flush();
    checkTest();

    setupTest();
    for(auto in : input) {
        ring.Push(in);
    }
    ring.Flush();
    checkTest();
    
    cout << "[\u2713] Ring processor module." << endl;

    const int count = 20;
    const size_t order = 3;
    int lastPushed = -1;
    int lastReceived = -1;

    auto checkDelay = [&lastPushed, &lastReceived, &count](int in) {
        if(lastPushed != count - 1) {
            AssertEQM(in, lastPushed - order, "Element is delayed");
        }

        AssertEQM(in, lastReceived + 1, "Element is delayed in correct order");
        lastReceived = in;
    };
    
    auto checkPush = [&lastPushed](int in) {
        AssertEQM(in, lastPushed, "Element is pushed");
    };

    QueueProcessor<int> queue(order, checkPush, checkDelay);  

    for(int i = 0; i < count; i++) {
        lastPushed = i;
        queue.Push(i);
    }

    queue.Flush();

    AssertEQM(lastReceived, count - 1, "All elements were delayed.");

    cout << "[\u2713] Queue processor module." << endl;
}
