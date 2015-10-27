#include <string>
#include <chrono>

#ifndef OPTONAUT_STATIC_TIMER_HEADER
#define OPTONAUT_STATIC_TIMER_HEADER

namespace optonaut {
    class STimer {
        private:
        std::chrono::high_resolution_clock::time_point last;
        public:
        STimer() : last(std::chrono::high_resolution_clock::now()) { }
        void Tick(std::string label = "");
    };
}

#endif
