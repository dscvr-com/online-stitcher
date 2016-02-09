#include <cassert>
#include <string>
#include <iostream>
#include <sstream>
#include <android/log.h>


//#include "backtrace.hpp"
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
#define AssertFalseInProduction(x) AssertFalseInProduction_(x, #x)

namespace optonaut {

    inline void PrintAndTerminate(std::string message, std::string vars, std::string values = "", bool isWarning = false) {
        std::stringstream s;
        if(isWarning) {
            s << "Warning: ";
        } else {
            s << "Assertion Failed: ";
        }
        if(message != "")
            s << message << " ";
        if(vars != "")
            s << std::endl << "Expression: " << vars << " ";
        if(values != "")
            s << "(" << values << ")";

        s << std::endl;

        std::string ret = s.str();
        __android_log_print(ANDROID_LOG_DEBUG, "TAG", "%s", ret.c_str());

        if(!isWarning) {
            //PrintBacktrace();
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

    #if __APPLE__
        #include "TargetConditionals.h"
        #if TARGET_OS_IPHONE
        inline void AssertFalseInProduction_(bool val, std::string var) {
            Assert_(!val, "", var, false);
        }
        #else
            inline void AssertFalseInProduction_(bool, std::string) { }
        #endif
    #else
        inline void AssertFalseInProduction_(bool, std::string) { }
    #endif
}
#endif
