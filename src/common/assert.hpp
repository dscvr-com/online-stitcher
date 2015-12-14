#include <cassert>
#include <string>
#include <iostream>

#include "backtrace.hpp"
#include "support.hpp"

#ifndef OPTONAUT_ASSERT_HEADER
#define OPTONAUT_ASSERT_HEADER

#define AssertEQ(x, y) AssertEQ_(x, y, "", #x" == "#y, false)
#define AssertEQM(x, y, msg) AssertEQ_(x, y, msg, #x" == "#y, false)
#define AssertNEQ(x, y) AssertNEQ_(x, y, "", #x" != "#y, false)
#define AssertNEQM(x, y, msg) AssertNEQ_(x, y, msg, #x" != "#y, false)
#define AssertGE(x, y) AssertGE_(x, y, "", #x" >= "#y, false)
#define AssertGEM(x, y, msg) AssertGE_(x, y, msg, #x" >= "#y, false)
#define AssertGT(x, y) AssertGT_(x, y, "", #x" > "#y, false)
#define AssertGTM(x, y, msg) AssertGT_(x, y, msg, #x" > "#y, false)
#define Assert(x) Assert_(x, "", #x, false)
#define AssertM(x, msg) Assert_(x, msg, #x, false)

#define AssertWEQ(x, y) AssertEQ_(x, y, "", #x" == "#y, true)
#define AssertWEQM(x, y, msg) AssertEQ_(x, y, msg, #x" == "#y, true)
#define AssertWNEQ(x, y) AssertNEQ_(x, y, "", #x" != "#y, true)
#define AssertWNEQM(x, y, msg) AssertNEQ_(x, y, msg, #x" != "#y, true)
#define AssertWGE(x, y) AssertGE_(x, y, "", #x" >= "#y, true)
#define AssertWGEM(x, y, msg) AssertGE_(x, y, msg, #x" >= "#y, true)
#define AssertWGT(x, y) AssertGT_(x, y, "", #x" > "#y, true)
#define AssertWGTM(x, y, msg) AssertGT_(x, y, msg, #x" > "#y, true)
#define AssertW(x) Assert_(x, "", #x, true)
#define AssertWM(x, msg) Assert_(x, msg, #x, true)

namespace optonaut {

    inline void PrintAndTerminate(std::string message, std::string vars, std::string values = "", bool isWarning = false) {

        if(isWarning) {
            std::cerr << "Warning: "; 
        } else {
            std::cerr << "Assertion Failed: ";
        }
        if(message != "")
            std::cerr << message << " ";
        if(vars != "")
            std::cerr << std::endl << "Expresssion: " << vars << " ";
        if(values != "")
            std::cerr << "(" << values << ")";

        std::cerr << std::endl;

        if(!isWarning) {
            PrintBacktrace();
            std::abort();
        }
    }
    inline void Assert_(bool condition, 
            std::string message, 
            std::string vars, 
            bool isWarning) {
        if(!condition) {
            PrintAndTerminate(message, vars, "", isWarning);
        }
    }

    template <typename T>
    inline void AssertEQ_(T a, T b, 
            std::string message, 
            std::string vars,
            bool isWarning,
            typename std::enable_if<!std::is_floating_point<T>::value >::type* = 0) {
        if(a != b) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " == " + ToString(b), isWarning);
        }
    }
    
    template <typename T>
    inline void AssertNEQ_(T a, T b, 
            std::string message, 
            std::string vars, 
            bool isWarning) {
        if(a == b) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " != " + ToString(b), isWarning);
        }
    }
    
    template <typename T>
    inline void AssertGT_(T a, T b, 
            std::string message, 
            std::string vars, 
            bool isWarning) {
        if(a <= b) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " > " + ToString(b), isWarning);
        }
    }
    
    template <typename T>
    inline void AssertGE_(T a, T b, 
            std::string message, 
            std::string vars,
            bool isWarning) {
        if(a < b) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " >= " + ToString(b), isWarning);
        }
    }
    
    template<typename T>
    inline void AssertEQ_(T a, T b, 
            std::string message, std::string vars,
            bool isWarning,
            typename std::enable_if<std::is_floating_point<T>::value >::type* = 0) {
        if(std::abs(a - b) > 0.00001) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " == " + ToString(b), isWarning);
        }
    }
    
    template<typename T>
    inline void AssertNEQ_(T a, T b, 
            std::string message, std::string vars,
            bool isWarning,
            typename std::enable_if<std::is_floating_point<T>::value >::type* = 0) {
        if(std::abs(a - b) < 0.00001) {
            PrintAndTerminate(message, vars, 
                    ToString(a) + " == " + ToString(b), isWarning);
        }
    }
}

#endif  
