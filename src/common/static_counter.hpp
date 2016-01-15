#include <string>
#include <map>

#ifndef OPTONAUT_STATIC_COUNTER_HEADER
#define OPTONAUT_STATIC_COUNTER_HEADER

namespace optonaut {
    class SCounter {
        private:
        int count;
        std::string label;
        public:
        SCounter(std::string label) : count(0), label(label) { } 
        void Increase();
        void Reset();
        ~SCounter();
    };
        
    class SCounters {
        private: 
            static std::map<std::string, std::shared_ptr<SCounter>> counters;
        public:
            static void Increase(std::string label);
    };
}

#endif
