#include <string>
#include <chrono>


#ifndef OPTONAUT_STATIC_TIMER_HEADER
#define OPTONAUT_STATIC_TIMER_HEADER

namespace optonaut {
    /*
     * Simple timer class. Can be used for debugging. 
     */
    class STimer {
        private:
        /*
         * Global enable/disable flag. 
         */
        static const bool g_enabled = true;

        bool enabled;
        std::chrono::high_resolution_clock::time_point last;

        public:
        /*
         * Creates a new instance of this class. 
         *
         * @param enable Forcefully enable/disable the timer. 
         */
        STimer(bool enabled = g_enabled) : enabled(enabled), last(std::chrono::high_resolution_clock::now()) { }

        /*
         * Measures past time since timer creation, timer reset or last tick event and 
         * outputs the measured time to the console.
         *
         * @param label The label to add to the output. 
         */
        void Tick(std::string label = "");

        /*
         * Resets the timer. 
         */
        void Reset();
    };
}

#endif
