#include "path.hpp"
namespace path {

fs::path _workspace = fs::path("/home/nvidia/git/mt-dist");
fs::path _res = _workspace / "res";
fs::path _models = _res / "models";
fs::path _ir = _res / "ir";
fs::path _out = _workspace / "out";

std::string join(const std::string head, const std::string tail) {
    if (tail.empty()) {
        return head;
    } else {
        return fs::path(head) / tail;
    }
}

std::string join(const fs::path head, const std::string tail = "") {
    if (tail.empty()) {
        return head;
    } else {
        return head / tail;
    }
}

std::string workspace(const std::string path) {
    return join(_workspace, path);
}
std::string res(const std::string path) {
    return join(_res, path);
}
std::string models(const std::string path) {
    return join(_models, path);
}
std::string ir(const std::string path) {
    return join(_ir, path);
}
std::string out(const std::string path) {
    return join(_out, path);
}
};  // namespace path