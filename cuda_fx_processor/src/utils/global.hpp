
#include <string>
#define errChk(ans) \
    { errChkAssert((ans), __FILE__, __LINE__); }
inline void errChkAssert(int code, const char *file, int line, bool abort = true) {
    if (code > 0) {
        fprintf(stderr, "ErrChkAssert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}