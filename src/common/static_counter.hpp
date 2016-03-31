#include <string>
#include <memory>
#include <map>

#ifndef OPTONAUT_STATIC_COUNTER_HEADER
#define OPTONAUT_STATIC_COUNTER_HEADER

namespace optonaut {
    /*
     * Static counter. Can be increased or reset during program run, 
     * will output it's data at destruction. 
     */
    class SCounter {
        private:
        int count;
        std::string label;
        public:
        /*
         * Creates a new instance of this class with the given label. 
         */
        SCounter(std::string label) : count(0), label(label) { } 

        /*
         * Increases the counter. 
         */
        void Increase();

        /*
         * Resets the counter to zero. 
         */
        void Reset();
        ~SCounter();
    };
       
    /*
     * Static registry of counters, identified by their name/label. 
     *
     * The data is printed on program termination. 
     */ 
    class SCounters {
        private: 
            static std::map<std::string, std::shared_ptr<SCounter>> counters;
        public:
            /*
             * Increases the counter with the given name.
             */
            static void Increase(std::string label);
    };
}

#endif
