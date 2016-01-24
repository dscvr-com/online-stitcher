#include <string>
#include <chrono>

#ifndef OPTONAUT_STATIC_TIMER_HEADER
#define OPTONAUT_STATIC_TIMER_HEADER

namespace optonaut {
    class STimer {
        private:
        static const bool g_enabled = false;

        bool enabled;
        std::chrono::high_resolution_clock::time_point last;

        public:
        STimer(bool enabled = g_enabled) : enabled(enabled), last(std::chrono::high_resolution_clock::now()) { }
        void Tick(std::string label = "");
        void Reset();
    };
}

#endif
