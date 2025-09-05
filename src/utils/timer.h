#pragma once
#include <chrono>
#include <string>
#include <iostream>

namespace chimera {

class Timer {
public:
    Timer(const std::string& name = "") : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        if (!stopped_) {
            stop();
        }
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();

        if (!name_.empty()) {
            std::cout << "⏱️  " << name_ << ": " << duration << " μs" << std::endl;
        }

        stopped_ = true;
    }

    long elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    bool stopped_ = false;
};

} // namespace chimera
