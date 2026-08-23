#include <tcapi/tcapi.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using TenT = tcapi::ItensorReal;
using Tensor = tcapi::ten_t<TenT>;
using Context = tcapi::context_handle_t<TenT>;

double exact_tfim_energy_per_site(double J, double g, long num_points = 200001)
{
    if(num_points < 2)
        throw std::invalid_argument("num_points must be at least 2");

    const double dk = 2.0 * M_PI / static_cast<double>(num_points - 1);
    double integral = 0.0;
    for(long n = 0; n < num_points; ++n)
    {
        const double k = n * dk;
        double w = 1.0;
        if(n == 0 || n == num_points - 1) w = 0.5;
        integral += w * std::sqrt(J * J + g * g - 2.0 * J * g * std::cos(k));
    }
    integral *= dk;
    return -(1.0 / (2.0 * M_PI)) * integral;
}

Tensor make_tfim_two_site_hamiltonian(Context& ctx, double J, double g)
{
    Tensor h = tcapi::zeros<TenT>(ctx, {4, 4});
    tcapi::set_elem<TenT>(ctx, h, {0, 0}, J);
    tcapi::set_elem<TenT>(ctx, h, {3, 3}, J);
    tcapi::set_elem<TenT>(ctx, h, {1, 1}, -J);
    tcapi::set_elem<TenT>(ctx, h, {2, 2}, -J);

    const double g2 = -0.5 * g;
    for(const long* c : (const long[8][2]){
        {0,1},{0,2},{1,0},{1,3},{2,0},{2,3},{3,1},{3,2}})
        tcapi::set_elem<TenT>(ctx, h, {c[0], c[1]}, g2);
    return h;
}

Tensor make_imaginary_time_gate(Context& ctx, const Tensor& h, double dt, long d = 2)
{
    tcapi::real_ten_t<TenT> lambda_mat;
    Tensor v;
    tcapi::eigh<TenT>(ctx, h, 1, lambda_mat, v);

    Tensor w_vector;
    tcapi::diag<TenT>(ctx, lambda_mat, w_vector);
    tcapi::for_each<TenT>(ctx, w_vector,
        [dt](double e) { return std::exp(-dt * e); });

    Tensor w_matrix;
    tcapi::diag<TenT>(ctx, w_vector, w_matrix);

    Tensor u;
    tcapi::contract<TenT>(ctx, v, "ij", w_matrix, "jk", u, "ik");
    Tensor v_transpose;
    tcapi::transpose<TenT>(ctx, v, {1, 0}, v_transpose);
    tcapi::contract<TenT>(ctx, u, "ij", v_transpose, "jk", u, "ik");

    Tensor gate;
    tcapi::reshape<TenT>(ctx, u, {d, d, d, d}, gate);
    return gate;
}

std::vector<double> itebd_imaginary_time(Context& ctx,
                                         const Tensor& gate,
                                         long N,
                                         double dt,
                                         long chi,
                                         double s_min = 1.0e-10,
                                         unsigned seed = 280711)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    auto gen = [&] { return dis(rng); };

    Tensor gamma[2] = {
        tcapi::random<TenT>(ctx, {chi, 2, chi}, gen),
        tcapi::random<TenT>(ctx, {chi, 2, chi}, gen),
    };
    Tensor lam[2] = {
        tcapi::random<TenT>(ctx, {chi}, gen),
        tcapi::random<TenT>(ctx, {chi}, gen),
    };

    std::vector<double> energies;

    for(long i = 0; i < N; ++i)
    {
        const long a = i % 2;
        const long b = (i + 1) % 2;

        Tensor lam_a_full;
        tcapi::diag<TenT>(ctx, lam[a], lam_a_full);
        Tensor lam_b_full;
        tcapi::diag<TenT>(ctx, lam[b], lam_b_full);

        Tensor theta;
        tcapi::contract<TenT>(ctx, lam_b_full, "ij", gamma[a], "jkl", theta, "ikl");
        tcapi::contract<TenT>(ctx, theta, "ijk", lam_a_full, "kl", theta, "ijl");
        tcapi::contract<TenT>(ctx, theta, "ijk", gamma[b], "klm", theta, "ijlm");
        tcapi::contract<TenT>(ctx, theta, "ijkl", lam_b_full, "lm", theta, "ijkm");
        tcapi::contract<TenT>(ctx, theta, "ijkl", gate, "jkmn", theta, "imnl");

        Tensor ga, gb;
        tcapi::real_ten_t<TenT> sigma;
        double trunc_err = 0.0;
        tcapi::trunc_svd<TenT>(ctx, theta, 2, ga, sigma, gb, trunc_err, chi, s_min);

        Tensor lam_a_work;
        tcapi::diag<TenT>(ctx, sigma, lam_a_work);
        tcapi::normalize<TenT>(ctx, lam_a_work);
        lam[a] = std::move(lam_a_work);

        Tensor lam_b_work = lam[b];
        tcapi::for_each<TenT>(ctx, lam_b_work,
            [](double e) { return 1.0 / e; });
        Tensor lam_b_inv;
        tcapi::diag<TenT>(ctx, lam_b_work, lam_b_inv);

        tcapi::contract<TenT>(ctx, lam_b_inv, "ij", ga, "jkl", ga, "ikl");
        tcapi::contract<TenT>(ctx, gb, "ijk", lam_b_inv, "kl", gb, "ijl");
        gamma[a] = std::move(ga);
        gamma[b] = std::move(gb);

        if(i >= 1)
        {
            double theta_norm_sq = 0.0;
            tcapi::for_each<TenT>(ctx, theta,
                [&](double e) { theta_norm_sq += e * e; });
            energies.push_back(-std::log(theta_norm_sq) / dt / 2.0);
        }
    }

    return energies;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const double J = argc > 1 ? std::stod(argv[1]) : 1.0;
        const double g = argc > 2 ? std::stod(argv[2]) : 0.5;
        const double dt = argc > 3 ? std::stod(argv[3]) : 0.005;
        const long chi = argc > 4 ? std::stol(argv[4]) : 5;
        const long N = argc > 5 ? std::stol(argv[5]) : 3000;
        if(!(dt > 0.0) || chi < 1 || N < 2)
            throw std::invalid_argument("usage: itebd [J] [g] [dt>0] [chi>=1] [N>=2]");

        Context ctx;
        tcapi::create_context(ctx);
        auto h = make_tfim_two_site_hamiltonian(ctx, J, g);
        auto gate = make_imaginary_time_gate(ctx, h, dt);
        auto energies = itebd_imaginary_time(ctx, gate, N, dt, chi);
        tcapi::destroy_context(ctx);

        const double exact = exact_tfim_energy_per_site(J, g);
        const double final_estimate = energies.back();
        const double abs_diff = std::abs(final_estimate - exact);
        auto last100 = std::min<size_t>(100, energies.size());
        double lo = *std::min_element(energies.end() - last100, energies.end());
        double hi = *std::max_element(energies.end() - last100, energies.end());

        std::cout << "Appendix A: TFIM imaginary-time iTEBD\n"
                  << "parameters: J=" << J << ", g=" << g << ", dt=" << dt
                  << ", chi=" << chi << ", N=" << N << '\n'
                  << "final E_iTEBD       = " << final_estimate << '\n'
                  << "exact energy/site   = " << exact << '\n'
                  << "absolute difference = " << abs_diff << '\n'
                  << "final 100-iteration span = " << (hi - lo) << '\n';
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
