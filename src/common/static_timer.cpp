#include <chrono>
#include <iostream>
#include "static_timer.hpp"

using namespace std;

namespace optonaut {

    static const bool enabled = false;

    void STimer::Tick(string label) {

        if(!enabled)
            return; 

        auto now = chrono::high_resolution_clock::now();
        auto duration = now - last;

        if(label != "") 
            cout << label << ": ";
        cout << chrono::duration_cast<chrono::milliseconds>(duration).count() << "ms" << endl;

        last = now;
    }

    void STimer::Reset() {
        last = chrono::high_resolution_clock::now();
    }
}
