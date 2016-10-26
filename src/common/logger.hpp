#include <string>
#include <iostream>
#include <sstream>
#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifndef OPTONAUT_LOGGER_HEADER
#define OPTONAUT_LOGGER_HEADER

#define Log Logger(__PRETTY_FUNCTION__, true)
#define LogR Logger("RESULT ", false)

namespace optonaut {
class Logger
{
    private:
        std::ostringstream line;
        inline std::string MethodName(const std::string& prettyFunction)
        {
            size_t end = prettyFunction.rfind("(");
            size_t begin = prettyFunction.substr(0,end).rfind("optonaut::") + 10;
            end = end - begin;

            return prettyFunction.substr(begin,end);
        }
    public:
        Logger(std::string function, bool isFunctionName) : line() {
            if(isFunctionName) {
                line << "[" << MethodName(function) << "] ";   
            } else {
                line << function;
            }
        };

        template <typename T>
        Logger& operator << (const T& t) {
           line << t;
           return *this;
        }

        ~Logger() {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_DEBUG, "OPTONAUT_ONLINE_STITCHER", "%s", line.str().c_str());
#else
            std::cout << line.str() << std::endl;
#endif
        }
};
}
#endif
