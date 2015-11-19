#include <cassert>
#include <string>
#include <iostream>

#include "backtrace.hpp"
#include "support.hpp"

#ifndef OPTONAUT_ASSERT_HEADER
#define OPTONAUT_ASSERT_HEADER

#define AssertEQ(x, y) AssertEQ_(x, y, "", #x" == "#y)
#define AssertEQM(x, y, msg) AssertEQ_(x, y, msg, #x" == "#y)
#define AssertGT(x, y) AssertGT_(x, y, "", #x" > "#y)
#define AssertGTM(x, y, msg) AssertGT_(x, y, msg, #x" > "#y)
#define Assert(x) Assert_(x, "", #x)
#define AssertM(x, msg) Assert_(x, msg, #x)

namespace optonaut {

    inline void PrintAndTerminate(std::string message, std::string vars, std::string values = "") {
        std::cerr << "Assertion Failed. " << std::endl;
        if(message != "")
            std::cerr << message << std::endl;
        if(vars != "")
            std::cerr << "(" << vars << ")" << std::endl;
        if(values != "")
            std::cerr << "(" << values << ")" << std::endl;
        PrintBacktrace();
        std::abort();
    }
    inline void Assert_(bool condition, std::string message, std::string vars) {
        if(!condition) {
            PrintAndTerminate(message, vars);
        }
    }

    template <typename T, typename V>
    inline void AssertEQ_(T a, V b, 
            std::string message, 
            std::string vars,
            typename std::enable_if<!std::is_floating_point<T>::value >::type* = 0, 
            typename std::enable_if<!std::is_floating_point<V>::value >::type* = 0) {
        if(a != (T)b) {
            PrintAndTerminate(message, vars, ToString(a) + " == " + ToString(b));
        }
    }
    
    template <typename T, typename V>
    inline void AssertGT_(T a, V b, std::string message, std::string vars) {
        if(a <= (T)b) {
            PrintAndTerminate(message, vars, ToString(a) + " < " + ToString(b));
        }
    }
    
    template<typename T, typename V>
    inline void AssertEQ_(T a, V b, 
            std::string message, std::string vars,
            typename std::enable_if<std::is_floating_point<T>::value >::type* = 0, 
            typename std::enable_if<std::is_floating_point<V>::value >::type* = 0) {
        if(std::abs(a - b) > 0.00001) {
            PrintAndTerminate(message, vars, ToString(a) + " == " + ToString(b));
        }
    }
}

#endif  
