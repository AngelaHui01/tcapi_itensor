#pragma once

#include "tcapi/tcapi.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace tcapi;

namespace tc_test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void report(const char* file, int line, const std::string& msg)
{
    ++g_failures;
    std::printf("[FAILED] %s:%d: %s\n", file, line, msg.c_str());
}

// shape comparison (shapes are element-type independent)
inline bool shape_equals(ItensorContext& ctx, const itensor::ITensor& t,
                         std::initializer_list<long> want)
{
    auto got = tcapi::shape<ItensorReal>(ctx, t);
    if(got.size() != want.size()) return false;
    std::size_t i = 0;
    for(long w : want)
        if(got[i++] != w) return false;
    return true;
}

// logical element access
template<typename TenT>
inline elem_t<TenT> at(const context_handle_t<TenT>& ctx,
                       const ten_t<TenT>& t,
                       std::initializer_list<long> c)
{
    return get_elem<TenT>(ctx, t, elem_coors_t<TenT>(c.begin(), c.end()));
}

// real / complex scalar comparison
inline bool approx(double a, double b, double tol = 1e-10)
{
    return std::abs(a - b) < tol;
}

inline bool approx(std::complex<double> a, std::complex<double> b, double tol = 1e-9)
{
    return std::abs(a - b) < tol;
}

// all-element relative/absolute tolerance check (shape must match)
template<typename TenT>
inline bool all_close(const context_handle_t<TenT>& ctx,
                      const ten_t<TenT>& a, const ten_t<TenT>& b,
                      double rtol = 1e-7, double atol = 1e-12)
{
    auto sa = shape<TenT>(ctx, a);
    auto sb = shape<TenT>(ctx, b);
    if(sa != sb) return false;
    bool ok = true;
    detail::for_each_coordinate<TenT>(sa, [&](const elem_coors_t<TenT>& c)
    {
        auto x = get_elem<TenT>(ctx, a, c);
        auto y = get_elem<TenT>(ctx, b, c);
        if(std::abs(x - y) > atol + rtol * std::abs(y)) ok = false;
    });
    return ok;
}

// Frobenius norm and max-abs helpers
template<typename TenT>
inline double frob(const context_handle_t<TenT>& ctx, const ten_t<TenT>& t)
{
    auto sh = shape<TenT>(ctx, t);
    double s = 0;
    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& c)
    {
        auto v = get_elem<TenT>(ctx, t, c);
        s += std::norm(v);
    });
    return std::sqrt(s);
}

template<typename TenT>
inline double max_abs(const context_handle_t<TenT>& ctx, const ten_t<TenT>& t)
{
    auto sh = shape<TenT>(ctx, t);
    double m = 0;
    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& c)
    {
        m = std::max(m, std::abs(get_elem<TenT>(ctx, t, c)));
    });
    return m;
}

// SVD reconstruction residual ||a - u*s*vdag|| (Frobenius)
template<typename TenT>
inline double svd_residual(const context_handle_t<TenT>& ctx,
                           const ten_t<TenT>& a,
                           const ten_t<TenT>& u,
                           const ten_t<TenT>& s,
                           const ten_t<TenT>& vdag)
{
    auto recon = contract<TenT>(ctx, contract<TenT>(ctx, u, s), vdag);
    auto sh = shape<TenT>(ctx, a);
    double err2 = 0;
    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& c)
    {
        auto d = get_elem<TenT>(ctx, a, c) - get_elem<TenT>(ctx, recon, c);
        err2 += std::norm(d);
    });
    return std::sqrt(err2);
}

// singular values (reads the diagonal of a real sigma tensor)
template<typename TenT>
inline std::vector<double> sing_vals(const context_handle_t<TenT>& ctx,
                                     const real_ten_t<TenT>& sigma)
{
    std::vector<double> out;
    auto sh = shape<TenT>(ctx, sigma);
    bond_dim_t<TenT> kappa = sh[0];
    for(bond_dim_t<TenT> k = 0; k < kappa; ++k)
        out.push_back(std::abs(at<TenT>(ctx, sigma, {k, k})));
    return out;
}

// eigenvalues from a diagonal matrix tensor (complex storage)
inline std::vector<std::complex<double>>
diag_elements_c(ItensorContext& ctx, const itensor::ITensor& L)
{
    std::vector<std::complex<double>> out;
    auto is = itensor::inds(L);
    long n = itensor::dim(is[0]);
    for(long k = 0; k < n; ++k)
        out.push_back(get_elem<ItensorCplx>(ctx, L, {k, k}));
    return out;
}

// eigenvalue multiset matching (sorted by real, then imaginary part)
inline bool match_eigenvalues(const std::vector<std::complex<double>>& got,
                              const std::vector<std::complex<double>>& want,
                              double rtol, double atol)
{
    if(got.size() != want.size()) return false;
    auto g = got, w = want;
    auto key = [](std::complex<double> z) { return std::make_pair(z.real(), z.imag()); };
    std::sort(g.begin(), g.end(), [&](auto const& x, auto const& y) { return key(x) < key(y); });
    std::sort(w.begin(), w.end(), [&](auto const& x, auto const& y) { return key(x) < key(y); });
    for(std::size_t i = 0; i < g.size(); ++i)
        if(std::abs(g[i] - w[i]) > atol + rtol * std::abs(w[i]))
            return false;
    return true;
}

// test runner
inline void run_test(const char* name, void (*fn)())
{
    std::printf("[ RUN  ] %s\n", name);
    int before = g_failures;
    fn();
    if(g_failures == before)
        std::printf("[  OK  ] %s\n", name);
    else
        std::printf("[FAILED] %s\n", name);
}

} // namespace tc_test

#define CHECK(cond) \
    do { ++tc_test::g_checks; if(!(cond)) tc_test::report(__FILE__, __LINE__, #cond); } while(0)

#define CHECK_APPROX(a, b, tol) \
    do { ++tc_test::g_checks; if(!tc_test::approx((a), (b), (tol))) \
             tc_test::report(__FILE__, __LINE__, "approx(" #a ", " #b ") failed"); } while(0)

#define CHECK_THROW(EXC_TYPE, EXPR) \
    do { \
        ++tc_test::g_checks; \
        bool _caught_ = false; \
        try { EXPR; } catch(const EXC_TYPE&) { _caught_ = true; } catch(...) {} \
        if(!_caught_) \
            tc_test::report(__FILE__, __LINE__, "expected " #EXC_TYPE " from: " #EXPR); \
    } while(0)

#define CHECK_SHAPE(ctx, t, ...) \
    do { ++tc_test::g_checks; \
         if(!tc_test::shape_equals((ctx), (t), {__VA_ARGS__})) \
             tc_test::report(__FILE__, __LINE__, "shape mismatch: " #__VA_ARGS__); } while(0)

#define CHECK_ALL_CLOSE(ctx, T, a, b, rtol, atol) \
    do { ++tc_test::g_checks; \
         if(!tc_test::all_close<T>((ctx), (a), (b), (rtol), (atol))) \
             tc_test::report(__FILE__, __LINE__, #a " != " #b " within tolerance"); } while(0)