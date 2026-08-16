// tcapi/diagnostics.h
// Mirrors tcapi_numpy/diagnostics.py.
// Sec. C2g — TCAPI_VERBOSE environment-variable diagnostics.
//   TCAPI_VERBOSE=0 (default): silent.
//   TCAPI_VERBOSE=1: one line per TCAPI call with function name + salient
//                    input info (order/shape/elem type).
//   TCAPI_VERBOSE=2: level-1 output plus measured wall-clock time.
//
// The public functions are templated, so the wrapper framework is kept out
// of the way: each exported wrapper (or its detail helper) calls
// tcapi::detail::verbose_guard, an RAII object that logs on construction and
// destruction.
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
    // Accept an integer. Anything non-parseable / negative behaves as 0.
    if(s.empty()) return 0;
    for(char c : s)
        if(c < '0' || c > '9') return 0;
    return static_cast<int>(s[0] - '0'); // leading digit only for level 0..9
}

/// RAII guard logging one TCAPI call according to TCAPI_VERBOSE.
/// Usage in a wrapper: auto _v = tcapi::detail::verbose_guard("contract", args...);
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
        // Fall back to a size-based generic summary where possible.
        std::cout << arg;
    }

    int level_ = 0;
    std::chrono::steady_clock::time_point start_;
};

} // namespace detail

} // namespace tcapi