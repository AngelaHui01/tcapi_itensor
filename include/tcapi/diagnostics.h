#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace tcapi {

namespace detail {

inline int verbose_level() noexcept
{
    const char* env = std::getenv("TCAPI_VERBOSE");
    if(env == nullptr) return 0;
    std::string s(env);
    if(s.empty()) return 0;
    for(char c : s)
        if(c < '0' || c > '9') return 0;
    return static_cast<int>(s[0] - '0'); // leading digit only for level 0..9
}

class verbose_guard
{
public:
    template<typename... Ts>
    verbose_guard(const char* name, const Ts&... args)
        : level_(verbose_level())
    {
        if(level_ == 0) return;
        std::cout << "[TCAPI] " << name;
        if(level_ >= 1 && sizeof...(Ts) > 0)
        {
            std::cout << "(";
            (append(args), ...);
            std::cout << ")";
        }
        if(level_ >= 2)
        {
            start_ = std::chrono::steady_clock::now();
            std::cout.flush();
        }
        else
        {
            std::cout << "\n";
        }
    }

    ~verbose_guard()
    {
        if(level_ == 0) return;
        if(level_ >= 2)
        {
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            std::cout << "  |  elapsed=" << ms << " ms\n";
        }
    }

private:
    template<typename T>
    void append(const T& arg) const
    {
        summary(arg);
    }

    static void summary(const int v)            { std::cout << v; }
    static void summary(const long v)           { std::cout << v; }
    static void summary(const double v)         { std::cout << v; }
    static void summary(const std::string& s)   { std::cout << s; }
    static void summary(const char* s)          { std::cout << (s ? s : "null"); }
    template<typename T>
    static void summary(const T& arg)
    {
        std::cout << arg;
    }

    int level_ = 0;
    std::chrono::steady_clock::time_point start_;
};

} // namespace detail

} // namespace tcapi