#include <tcapi/tcapi.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using TenT = tcapi::ItensorReal;
using Tensor = tcapi::ten_t<TenT>;
using Context = tcapi::context_handle_t<TenT>;

struct Factors
{
    Tensor left;
    Tensor right;
    double truncation_error = 0.0;
};

Factors split_factor(const Context& ctx, const Tensor& a, int row_bonds, long maxdim)
{
    Tensor u;
    Tensor vdag;
    tcapi::real_ten_t<TenT> sigma;
    double truncation_error = 0.0;
    tcapi::trunc_svd<TenT>(ctx, a, row_bonds, u, sigma, vdag,
                            truncation_error, maxdim, 0.0);

    auto u_shape = tcapi::shape<TenT>(ctx, u);
    auto v_shape = tcapi::shape<TenT>(ctx, vdag);
    if(u_shape.size() != 3 || v_shape.size() != 3 || u_shape[2] != v_shape[0])
        throw std::runtime_error("TRG: unexpected SVD factor shapes.");

    Tensor left = tcapi::allocate<TenT>(ctx, u_shape);
    Tensor right = tcapi::allocate<TenT>(ctx, v_shape);
    for(long i = 0; i < u_shape[0]; ++i)
        for(long j = 0; j < u_shape[1]; ++j)
            for(long k = 0; k < u_shape[2]; ++k)
            {
                auto root_sigma = std::sqrt(std::max(0.0,
                    tcapi::get_elem<TenT>(ctx, sigma, {k, k})));
                tcapi::set_elem<TenT>(ctx, left, {i, j, k},
                                      root_sigma * tcapi::get_elem<TenT>(ctx, u, {i, j, k}));
            }

    for(long k = 0; k < v_shape[0]; ++k)
        for(long i = 0; i < v_shape[1]; ++i)
            for(long j = 0; j < v_shape[2]; ++j)
            {
                auto root_sigma = std::sqrt(std::max(0.0,
                    tcapi::get_elem<TenT>(ctx, sigma, {k, k})));
                tcapi::set_elem<TenT>(ctx, right, {k, i, j},
                                      root_sigma * tcapi::get_elem<TenT>(ctx, vdag, {k, i, j}));
            }

    return {std::move(left), std::move(right), truncation_error};
}

Tensor ising_weight(const Context& ctx, double temperature)
{
    auto a = tcapi::zeros<TenT>(ctx, {2, 2, 2, 2});
    auto spin = [](long s) { return s == 0 ? 1.0 : -1.0; };
    for(long l = 0; l < 2; ++l)
        for(long r = 0; r < 2; ++r)
            for(long u = 0; u < 2; ++u)
                for(long d = 0; d < 2; ++d)
                {
                    auto energy = spin(l) * spin(d) + spin(d) * spin(r)
                                + spin(r) * spin(u) + spin(u) * spin(l);
                    tcapi::set_elem<TenT>(ctx, a, {l, r, u, d},
                                          std::exp(-energy / temperature));
                }
    return a;
}

double double_trace(const Context& ctx, const Tensor& a)
{
    auto dims = tcapi::shape<TenT>(ctx, a);
    if(dims.size() != 4 || dims[0] != dims[1] || dims[2] != dims[3])
        throw std::runtime_error("TRG: tensor has invalid double-trace shape.");

    double result = 0.0;
    for(long l = 0; l < dims[0]; ++l)
        for(long u = 0; u < dims[2]; ++u)
            result += tcapi::get_elem<TenT>(ctx, a, {l, l, u, u});
    return result;
}

double run_trg(const Context& ctx, double temperature, long maxdim, int topscale)
{
    Tensor a = ising_weight(ctx, temperature);
    double log_z_per_site = 0.0;

    for(int scale = 1; scale <= topscale; ++scale)
    {
        Tensor a_lurd;
        tcapi::transpose<TenT>(ctx, a, {0, 2, 1, 3}, a_lurd);
        auto left_right = split_factor(ctx, a_lurd, 2, maxdim);

        Tensor a_urld;
        tcapi::transpose<TenT>(ctx, a, {2, 1, 0, 3}, a_urld);
        auto up_down = split_factor(ctx, a_urld, 2, maxdim);

        Tensor upper;
        tcapi::contract<TenT>(ctx, left_right.left, "luL",
                               up_down.left, "urU", upper, "lLrU");
        Tensor lower;
        tcapi::contract<TenT>(ctx, upper, "lLrU",
                               left_right.right, "Rrd", lower, "lLURd");
        tcapi::contract<TenT>(ctx, lower, "lLURd",
                               up_down.right, "Dld", a, "LRUD");

        auto trace = double_trace(ctx, a);
        if(!(trace > 0.0) || !std::isfinite(trace))
            throw std::runtime_error("TRG: non-positive or non-finite normalization.");
        tcapi::scale<TenT>(ctx, a, 1.0 / trace);
        log_z_per_site += std::log(trace) / std::pow(2.0, 1.0 + scale);

        std::cout << "scale " << scale
                  << ": bond dimensions " << tcapi::shape<TenT>(ctx, a)[0]
                  << " x " << tcapi::shape<TenT>(ctx, a)[2]
                  << ", truncation errors " << left_right.truncation_error
                  << ", " << up_down.truncation_error << '\n';
    }

    return log_z_per_site;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const double temperature = argc > 1 ? std::stod(argv[1]) : 3.0;
        const long maxdim = argc > 2 ? std::stol(argv[2]) : 8;
        const int topscale = argc > 3 ? std::stoi(argv[3]) : 6;
        if(!(temperature > 0.0) || maxdim < 1 || topscale < 1)
            throw std::invalid_argument("usage: trg [temperature>0] [maxdim>=1] [topscale>=1]");

        Context ctx;
        tcapi::create_context(ctx);
        auto log_z_per_site = run_trg(ctx, temperature, maxdim, topscale);
        tcapi::destroy_context(ctx);

        std::cout << "T = " << temperature << ", maxdim = " << maxdim
                  << ", topscale = " << topscale << '\n'
                  << "log(Z)/N = " << log_z_per_site << '\n';
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
