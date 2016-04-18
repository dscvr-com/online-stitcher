#include <cassert>
#include <string>
#include <iostream>
#include <sstream>
#ifdef __ANDROID__
#include <android/log.h>
#endif


//#include "backtrace.hpp"
#include "support.hpp"

#ifndef OPTONAUT_ASSERT_HEADER
#define OPTONAUT_ASSERT_HEADER

/*
 * Assert macros. Format is always Assert[OPERATION][MESSAGE].
 * Operation defines the operation (Equal, Not Equal, Greater Equal, Greater Than).
 * Message flag (M) defines wether a message is added to the assert. 
 *
 * Usage examples: AssertEQ(1, 3), AssertEQM(1, 3, "1 and 3 are equal"), AssertGT(3, 2), AssertM(false, "Fails always)
 */
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

/*
 * As above, but only output a warning instead of terminating the program. 
 */
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

    /*
     * Prints a stack trace and terminates the program, if isWarning is set to false. 
     *
     * @param message The message to print. 
     * @param vars A string containing all printable variables.
     * @param values A string containing all printable values.  
     * @param isWarning Indicates wether this call should terminate the program or not. 
     */
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

#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_DEBUG, "OPTONAUT_ONLINE_STITCHER", "%s", ret.c_str());
#else
        std::cout << ret;
#endif

        if(!isWarning) {
            //PrintBacktrace();
            std::abort();
        }
    }

    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
    inline void Assert_(bool condition, 
            std::string message, 
            std::string vars, 
            bool isWarning) {
        if(!condition) {
            PrintAndTerminate(message, vars, "", isWarning);
        }
    }

    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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
    
    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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
    
    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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
    
    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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
    
    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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
    
    /*
     * Simple assert, signature is similar to PrintAndTerminate.  
     */
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

    /*
     * Assert that fails whenever built in a production environment (e.g. the app). 
     * This is used to check if we forgot any active debug flags. 
     */
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
