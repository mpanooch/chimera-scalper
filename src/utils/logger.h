#pragma once
#include <string>
#include <iostream>
#include <mutex>

namespace chimera {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string level_str;
        std::string color;

        switch(level) {
            case LogLevel::DEBUG:    level_str = "DEBUG"; color = "\033[36m"; break;
            case LogLevel::INFO:     level_str = "INFO "; color = "\033[32m"; break;
            case LogLevel::WARNING:  level_str = "WARN "; color = "\033[33m"; break;
            case LogLevel::ERROR:    level_str = "ERROR"; color = "\033[31m"; break;
            case LogLevel::CRITICAL: level_str = "CRIT "; color = "\033[35m"; break;
        }

        std::cout << color << "[" << level_str << "] " << message << "\033[0m" << std::endl;
    }

private:
    Logger() = default;
    std::mutex mutex_;
};

#define LOG_DEBUG(msg) chimera::Logger::instance().log(chimera::LogLevel::DEBUG, msg)
#define LOG_INFO(msg) chimera::Logger::instance().log(chimera::LogLevel::INFO, msg)
#define LOG_WARNING(msg) chimera::Logger::instance().log(chimera::LogLevel::WARNING, msg)
#define LOG_ERROR(msg) chimera::Logger::instance().log(chimera::LogLevel::ERROR, msg)
#define LOG_CRITICAL(msg) chimera::Logger::instance().log(chimera::LogLevel::CRITICAL, msg)

} // namespace chimera
