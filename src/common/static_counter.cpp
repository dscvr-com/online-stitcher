#include <chrono>
#include <iostream>
#include "static_counter.hpp"

using namespace std;

namespace optonaut {
    static const bool enabled = true;
    static const bool verbose = false;

    void SCounter::Increase() {
        if(!enabled)
            return;

        count++;
    }
    
    void SCounter::Reset() {
        count = 0;
    }

    SCounter::~SCounter() {
        if(!enabled)
            return;

        cout << label << ": " << count << endl;
    }
    
    std::map<std::string, std::shared_ptr<SCounter>> SCounters::counters;
    
    void SCounters::Increase(std::string label) {

        auto it = counters.find(label);

        if(it == counters.end()) {
            counters.insert(std::make_pair(
                        label, 
                        std::make_shared<SCounter>(label)
                        ));
        }

        if(verbose) {
            cout << label << endl;
        }

        counters.at(label)->Increase();
    }
}
