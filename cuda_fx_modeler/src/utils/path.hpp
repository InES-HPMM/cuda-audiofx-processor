#pragma once
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
namespace path {
std::string join(const std::string head, const std::string tail = "");
std::string workspace(const std::string path = "");
std::string res(const std::string path = "");
std::string ir(const std::string path = "");
std::string models(const std::string path = "");
std::string out(const std::string path = "");
};  // namespace path