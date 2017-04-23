//
// Created by Emi on 10/12/2016.
//
#ifdef __ANDROID__
#include <jni.h>

#ifndef APP_ANDROID_JNIHELPER_HPP
#define APP_ANDROID_JNIHELPER_HPP

namespace optonaut {
class JniHelper {
public:
    static JNIEnv *jni_context;
};
}

#endif //APP_ANDROID_JNIHELPER_HPP
#endif