#include <vector>
#include <random>
#include <chrono>
#include <thread>

#include "../math/stat.hpp"
#include "../common/assert.hpp"
#include "../common/asyncQueueWorker.hpp"

using namespace std;
using namespace optonaut;

void SleepRandom() {
    std::mt19937_64 eng{std::random_device{}()};  
    std::uniform_int_distribution<> dist{1, 100};
    std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});
}

int main(int, char**) {

    int lastReceived = -1;
    int count = 10;

    AsyncQueue<int> worker([&lastReceived] (int in) {
                AssertEQM(in, lastReceived + 1, "Receive happens in order");
                lastReceived += 1;
            });

    for(int i = 0; i < count; i++) {
        SleepRandom();
        worker.Push(i);
    }

    worker.Finish();

    AssertEQM(lastReceived, count - 1, "All messages were received");

    cout << "[\u2713] AsyncQueue module." << endl;
}
