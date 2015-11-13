#include <csignal>
#include <execinfo.h>

void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    cout << "Crash (" << sig << ") Stack Trace: " << endl;
    backtrace_symbols_fd(array, size, STDOUT_FILENO);
    exit(1);
}

void RegisterCrashHandler() {
    signal(SIGSEGV, handler);   // install our handler
    signal(SIGABRT, handler);   // install our handler
}
