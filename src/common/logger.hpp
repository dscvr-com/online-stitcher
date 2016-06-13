#include <string>
#include <iostream>
#include <sstream>
#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifndef OPTONAUT_LOGGER_HEADER
#define OPTONAUT_LOGGER_HEADER

#define Log Logger(__func__, true)
#define LogR Logger("RESULT ", false)

namespace optonaut {
class Logger
{
    private:
        std::ostringstream line;
    public:
        Logger(std::string function, bool isFunctionName) {
            if(isFunctionName) {
                line << "[" << function << "] ";   
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
