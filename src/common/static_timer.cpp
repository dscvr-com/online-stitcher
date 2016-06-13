#include <chrono>
#include <iostream>
#include "logger.hpp"
#include "static_timer.hpp"

using namespace std;

namespace optonaut {
    void STimer::Tick(string label) {

        if(!enabled)
            return; 

        auto now = chrono::high_resolution_clock::now();
        auto duration = now - last;

        Logger log("TIMING ", false);

        if(label != "") 
            log << label << ": ";
        log << chrono::duration_cast<chrono::milliseconds>(duration).count() << "ms";

        last = now;
    }

    void STimer::Reset() {
        last = chrono::high_resolution_clock::now();
    }
}
