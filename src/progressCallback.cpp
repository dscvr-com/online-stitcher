#include "progressCallback.hpp"

namespace optonaut {
    bool EmptyCallback(float) { return true; }
    ProgressCallback ProgressCallback::Empty = ProgressCallback(EmptyCallback);
}
