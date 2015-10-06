#include <chrono>
#include <iostream>
#include "static_timer.hpp"

using namespace std;

namespace optonaut {
    
    chrono::high_resolution_clock::time_point last = chrono::high_resolution_clock::now();

    void STimer::Tick(string label) {
        auto now = chrono::high_resolution_clock::now();
        auto duration = now - last;

        if(label != "") 
            cout << label << ": ";
        cout << chrono::duration_cast<chrono::milliseconds>(duration).count() << "ms" << endl;

        last = now;
        
    }
}
