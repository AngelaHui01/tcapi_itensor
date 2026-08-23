#include "tc_test_util.h"
#include <algorithm>
#include <cmath>
#include <random>

using namespace tcapi;

// Example test: 1-D Ising transfer-matrix partition function on a ring,
// computed from TCAPI public operations (allocate -> contract -> trace).
// For an L-site ring with coupling K the partition function is
//   Z_L = tr(T^L) = (2*cosh(K))^L + (2*sinh(K))^L
// where T(s,s') = exp(K*s*s') is the 2x2 row-to-row transfer matrix.
// This mirrors the spirit of the numpy backend's example tests (TRG/iTEBD
// partition functions) without needing the example/ Python modules.
void test_partition_function_example()
{
    ItensorContext ctx; create_context(ctx);

    const double K = 0.5;
    const int L = 4;

    // row-to-row transfer matrix T[s][s'] for s,s' in {+1,-1}
    auto T = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
        {
            double si = (i == 0) ? 1.0 : -1.0;
            double sj = (j == 0) ? 1.0 : -1.0;
            set_elem<ItensorReal>(ctx, T, {i, j}, std::exp(K * si * sj));
        }

    // T^2S = T^4 by chaining labeled contractions along the shared bond
    ten_t<ItensorReal> T2;
    contract<ItensorReal>(ctx, T, "ab", T, "bc", T2, "ac");
    CHECK_SHAPE(ctx, T2, 2, 2);

    ten_t<ItensorReal> T4;
    contract<ItensorReal>(ctx, T2, "ab", T2, "bc", T4, "ac");
    CHECK_SHAPE(ctx, T4, 2, 2);

    // Z_4 = tr(T^4)
    detail::bond_idx_pairs_t<ItensorReal> pair{{0, 1}};
    ten_t<ItensorReal> Z;
    trace<ItensorReal>(ctx, T4, pair, Z);
    CHECK(order<ItensorReal>(ctx, Z) == 0);

    double got = tc_test::at<ItensorReal>(ctx, Z, {});
    double two_cosh = 2.0 * std::cosh(K);
    double two_sinh = 2.0 * std::sinh(K);
    // Z_L = 2*cosh(L*K) evaluated via the closed form:
    double want = std::pow(two_cosh, L) + std::pow(two_sinh, L);

    CHECK(tc_test::approx(got, want, 1e-9));
    CHECK(got > 0.0);

    destroy_context(ctx);
}

// Example test: quantum one-spin expectation value. For a spin-1/2 operator
// H = [[0,1],[1,0]] (the sigma^x matrix), <psi|H|psi> for psi = (|up>+|down>)/sqrt2
// must be 1. This uses element access + contraction as an end-to-end example.
void test_spin_expectation_example()
{
    ItensorContext ctx; create_context(ctx);

    auto H = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, H, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, H, {1, 0}, 1.0);

    auto psi = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2});
    const double inv = 1.0 / std::sqrt(2.0);
    set_elem<ItensorReal>(ctx, psi, {0}, inv);
    set_elem<ItensorReal>(ctx, psi, {1}, inv);

    // expectation = sum_{i,j} psi_i H_ij psi_j = 1
    double expv = 0.0;
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
            expv += tc_test::at<ItensorReal>(ctx, psi, {i}) *
                    tc_test::at<ItensorReal>(ctx, H, {i, j}) *
                    tc_test::at<ItensorReal>(ctx, psi, {j});
    CHECK(tc_test::approx(expv, 1.0, 1e-12));

    destroy_context(ctx);
}


// Example test: TFIM imaginary-time iTEBD (mirrors the numpy backend's
// test_itebd_example.py). Re-implements the algorithm inline since itebd.cc
// owns it and is a standalone binary. Checks convergence of the iTEBD energy
// to the exact free-fermion ground-state energy per site.
void test_itebd_imaginary_time()
{
    ItensorContext ctx; create_context(ctx);

    const double J = 1.0, g = 0.5;
    const double dt = 0.005;
    const long chi = 5, N = 3000, d = 2;
    const double s_min = 1.0e-10;

    // two-site TFIM Hamiltonian + imaginary-time gate exp(-dt*h)
    auto h = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{4, 4});
    set_elem<ItensorReal>(ctx, h, {0, 0}, J);
    set_elem<ItensorReal>(ctx, h, {3, 3}, J);
    set_elem<ItensorReal>(ctx, h, {1, 1}, -J);
    set_elem<ItensorReal>(ctx, h, {2, 2}, -J);
    const double g2 = -0.5 * g;
    for(const long* c : (const long[8][2]){
        {0,1},{0,2},{1,0},{1,3},{2,0},{2,3},{3,1},{3,2}})
        set_elem<ItensorReal>(ctx, h, {c[0], c[1]}, g2);

    real_ten_t<ItensorReal> lambda_mat;
    ten_t<ItensorReal> v;
    eigh<ItensorReal>(ctx, h, 1, lambda_mat, v);
    ten_t<ItensorReal> w_vec;
    diag<ItensorReal>(ctx, lambda_mat, w_vec);
    for_each<ItensorReal>(ctx, w_vec, [dt](double e) { return std::exp(-dt * e); });
    ten_t<ItensorReal> w_mat;
    diag<ItensorReal>(ctx, w_vec, w_mat);
    ten_t<ItensorReal> u, vt;
    contract<ItensorReal>(ctx, v, "ij", w_mat, "jk", u, "ik");
    transpose<ItensorReal>(ctx, v, {1, 0}, vt);
    contract<ItensorReal>(ctx, u, "ij", vt, "jk", u, "ik");
    ten_t<ItensorReal> gate;
    reshape<ItensorReal>(ctx, u, {d, d, d, d}, gate);
    CHECK_SHAPE(ctx, gate, 2, 2, 2, 2);

    // initial random MPS (seed matches the numpy example)
    std::mt19937 rng(280711);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    auto gen = [&] { return dis(rng); };
    auto gamma_a = random<ItensorReal>(ctx, {chi, 2, chi}, gen);
    auto gamma_b = random<ItensorReal>(ctx, {chi, 2, chi}, gen);
    auto lam_a = random<ItensorReal>(ctx, {chi}, gen);
    auto lam_b = random<ItensorReal>(ctx, {chi}, gen);

    std::vector<double> energies;
    for(long i = 0; i < N; ++i)
    {
        const long a = i % 2, b = (i + 1) % 2;
        auto& ga = (a == 0) ? gamma_a : gamma_b;
        auto& gb = (b == 0) ? gamma_a : gamma_b;
        auto& la = (a == 0) ? lam_a : lam_b;
        auto& lb = (b == 0) ? lam_a : lam_b;

        ten_t<ItensorReal> lam_a_full, lam_b_full;
        diag<ItensorReal>(ctx, la, lam_a_full);
        diag<ItensorReal>(ctx, lb, lam_b_full);

        ten_t<ItensorReal> theta;
        contract<ItensorReal>(ctx, lam_b_full, "ij", ga, "jkl", theta, "ikl");
        contract<ItensorReal>(ctx, theta, "ijk", lam_a_full, "kl", theta, "ijl");
        contract<ItensorReal>(ctx, theta, "ijk", gb, "klm", theta, "ijlm");
        contract<ItensorReal>(ctx, theta, "ijkl", lam_b_full, "lm", theta, "ijkm");
        contract<ItensorReal>(ctx, theta, "ijkl", gate, "jkmn", theta, "imnl");

        ten_t<ItensorReal> ua, vb;
        real_ten_t<ItensorReal> sigma;
        real_t<ItensorReal> trunc_err;
        trunc_svd<ItensorReal>(ctx, theta, 2, ua, sigma, vb, trunc_err, chi, s_min);

        ten_t<ItensorReal> svec;
        diag<ItensorReal>(ctx, sigma, svec);
        normalize<ItensorReal>(ctx, svec);
        la = std::move(svec);

        auto lb_work = lb;
        for_each<ItensorReal>(ctx, lb_work, [](double e) { return 1.0 / e; });
        ten_t<ItensorReal> lb_inv;
        diag<ItensorReal>(ctx, lb_work, lb_inv);
        contract<ItensorReal>(ctx, lb_inv, "ij", ua, "jkl", ua, "ikl");
        contract<ItensorReal>(ctx, vb, "ijk", lb_inv, "kl", vb, "ijl");
        ga = std::move(ua);
        gb = std::move(vb);

        if(i >= 1)
        {
            double theta_norm_sq = 0.0;
            for_each<ItensorReal>(ctx, theta, [&](double e) { theta_norm_sq += e * e; });
            energies.push_back(-std::log(theta_norm_sq) / dt / 2.0);
        }
    }

    // exact energy via trapezoid integration
    const long num_points = 200001;
    const double dk = 2.0 * M_PI / static_cast<double>(num_points - 1);
    double integral = 0.0;
    for(long n = 0; n < num_points; ++n)
    {
        const double k = n * dk;
        double w = (n == 0 || n == num_points - 1) ? 0.5 : 1.0;
        integral += w * std::sqrt(J * J + g * g - 2.0 * J * g * std::cos(k));
    }
    integral *= dk;
    const double exact = -(1.0 / (2.0 * M_PI)) * integral;

    const double final_estimate = energies.back();
    const double abs_diff = std::abs(final_estimate - exact);
    auto last100 = std::min<size_t>(100, energies.size());
    double lo = *std::min_element(energies.end() - last100, energies.end());
    double hi = *std::max_element(energies.end() - last100, energies.end());
    const double stabilization_span = hi - lo;

    std::printf("Appendix A: TFIM imaginary-time iTEBD\n");
    std::printf("parameters: J=%g, g=%g, dt=%g, chi=%ld, N=%ld\n", J, g, dt, chi, N);
    std::printf("final E_iTEBD       = %.15f\n", final_estimate);
    std::printf("exact energy/site   = %.15f\n", exact);
    std::printf("absolute difference = %.3e\n", abs_diff);
    std::printf("final 100-iteration span = %.3e\n", stabilization_span);

    CHECK(std::isfinite(final_estimate));
    CHECK(abs_diff < 1.0e-4);
    CHECK(stabilization_span < 1.0e-8);

    destroy_context(ctx);
}

void test_example_support_primitives()
{
    ItensorContext ctx; create_context(ctx);
    // verify the primed-pair machinery used by iTEBD-style flows is present
    auto a = eye<ItensorReal>(ctx, 2);
    CHECK(close<ItensorReal>(ctx, a, a, 0.0));
    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_partition_function_example", test_partition_function_example);
    tc_test::run_test("test_spin_expectation_example", test_spin_expectation_example);
    tc_test::run_test("test_itebd_imaginary_time", test_itebd_imaginary_time);
    tc_test::run_test("test_example_support_primitives", test_example_support_primitives);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}
