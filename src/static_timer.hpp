#include <string>

#ifndef OPTONAUT_STATIC_TIMER_HEADER
#define OPTONAUT_STATIC_TIMER_HEADER

namespace optonaut {
    class STimer {
        public: 
        static void Tick(std::string label = "");
    };
}

#endif
