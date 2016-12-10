//
// Created by Emi on 10/12/2016.
//

#include "JniHelper.hpp"

#ifdef __ANDROID__
namespace optonaut {
    JNIEnv *JniHelper::jni_context;
}
#endif