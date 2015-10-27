#include <chrono>
#include <iostream>
#include "static_timer.hpp"

using namespace std;

namespace optonaut {

    void STimer::Tick(string label) {
        auto now = chrono::high_resolution_clock::now();
        auto duration = now - last;

        if(label != "") 
            cout << label << ": ";
        cout << chrono::duration_cast<chrono::milliseconds>(duration).count() << "ms" << endl;

        last = now;
        
    }
}
