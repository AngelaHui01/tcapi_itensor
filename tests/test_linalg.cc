#include "tc_test_util.h"
#include <complex>
#include <random>

using namespace tcapi;

static ten_t<ItensorReal> rnd(ItensorContext& ctx,
                              const shape_t<ItensorReal>& shape,
                              std::mt19937& engine)
{
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };
    return random<ItensorReal>(ctx, shape, gen);
}

// norm: reals and complexes, unprimed and primed-pair
void test_norm()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    CHECK(tc_test::approx(norm<ItensorReal>(ctx, a), std::sqrt(3.0), 1e-12));

    auto z = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{3, 2, 4});
    CHECK(tc_test::approx(norm<ItensorReal>(ctx, z), 0.0, 1e-12));

    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1.0, 1.0));
    CHECK(tc_test::approx(norm<ItensorCplx>(ctx, c), std::sqrt(8.0), 1e-12));

    destroy_context(ctx);
}

// diag on 1st- and 2nd-order tensors, real and complex
void test_diag()
{
    ItensorContext ctx; create_context(ctx);

    auto vec = fill<ItensorReal>(ctx, shape_t<ItensorReal>{3}, 2.0);
    ten_t<ItensorReal> d;
    diag<ItensorReal>(ctx, vec, d);
    CHECK_SHAPE(ctx, d, 3, 3);
    CHECK(tc_test::at<ItensorReal>(ctx, d, {1, 1}) == 2.0);
    CHECK(tc_test::at<ItensorReal>(ctx, d, {0, 1}) == 0.0);

    auto mat = eye<ItensorReal>(ctx, 3);
    ten_t<ItensorReal> di;
    diag<ItensorReal>(ctx, mat, di);
    CHECK(order<ItensorReal>(ctx, di) == 1);
    CHECK_SHAPE(ctx, di, 3);
    CHECK(tc_test::at<ItensorReal>(ctx, di, {0}) == 1.0);

    auto c = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, c, {0, 0}, std::complex<double>(1, 2));
    set_elem<ItensorCplx>(ctx, c, {1, 1}, std::complex<double>(3, -1));
    ten_t<ItensorCplx> cd;
    diag<ItensorCplx>(ctx, c, cd);
    CHECK(order<ItensorCplx>(ctx, cd) == 1);
    CHECK(tc_test::at<ItensorCplx>(ctx, cd, {0}) == std::complex<double>(1, 2));
    CHECK(tc_test::at<ItensorCplx>(ctx, cd, {1}) == std::complex<double>(3, -1));

    destroy_context(ctx);
}

// scale: in-place, out-of-place (source untouched), complex
void test_scale()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 2.0);
    scale<ItensorReal>(ctx, a, 3.0);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 0}) == 6.0);

    auto src = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 2.0);
    ten_t<ItensorReal> b;
    scale<ItensorReal>(ctx, src, 5.0, b);
    CHECK(tc_test::at<ItensorReal>(ctx, b, {1, 1}) == 10.0);
    CHECK(tc_test::at<ItensorReal>(ctx, src, {1, 1}) == 2.0);

    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(1.0, 1.0));
    scale<ItensorCplx>(ctx, c, std::complex<double>(0.0, 1.0));
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {0}) == std::complex<double>(-1.0, 1.0));

    destroy_context(ctx);
}

// normalize: returns old norm; result has unit norm; complex
void test_normalize()
{
    ItensorContext ctx; create_context(ctx);

    auto c = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 3.0);
    double n = normalize<ItensorReal>(ctx, c);
    CHECK(tc_test::approx(n, std::sqrt(4 * 9.0), 1e-12));
    CHECK(tc_test::approx(norm<ItensorReal>(ctx, c), 1.0, 1e-12));

    auto z = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(1.0, 1.0));
    double nc = normalize<ItensorCplx>(ctx, z);
    CHECK(tc_test::approx(nc, std::sqrt(4.0), 1e-12));
    CHECK(tc_test::approx(norm<ItensorCplx>(ctx, z), 1.0, 1e-12));

    destroy_context(ctx);
}

// trace: partial trace keeping a 2-bond block; also a full trace to scalar
void test_trace()
{
    ItensorContext ctx; create_context(ctx);

    // build t[i,j,p,q] = id[i,j] * id[p,q]
    auto id = eye<ItensorReal>(ctx, 3);
    auto t = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{3, 3, 3, 3});
    detail::for_each_coordinate<ItensorReal>(shape_t<ItensorReal>{3, 3, 3, 3},
        [&](const elem_coors_t<ItensorReal>& c)
        {
            double v = tc_test::at<ItensorReal>(ctx, id, {c[0], c[1]}) *
                       tc_test::at<ItensorReal>(ctx, id, {c[2], c[3]});
            detail::set_elem_impl<ItensorReal>(t, detail::to_ivs<ItensorReal>(itensor::inds(t), c), v);
        });

    // trace bonds (2,3): sum_q delta_{p,q} -> factor 3, keep bonds 0,1 => 3*id
    detail::bond_idx_pairs_t<ItensorReal> pairs{{2, 3}};
    ten_t<ItensorReal> out;
    trace<ItensorReal>(ctx, t, pairs, out);
    CHECK_SHAPE(ctx, out, 3, 3);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, out, {0, 0}), 3.0));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, out, {0, 1}), 0.0));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, out, {2, 2}), 3.0));

    // full trace of eye(4): sum_i id[i,i] = 4 -> scalar
    auto m = eye<ItensorReal>(ctx, 4);
    detail::bond_idx_pairs_t<ItensorReal> full{{0, 1}};
    ten_t<ItensorReal> tr;
    trace<ItensorReal>(ctx, m, full, tr);
    CHECK(order<ItensorReal>(ctx, tr) == 0);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, tr, {}), 4.0));

    // complex partial trace
    auto cm = eye<ItensorCplx>(ctx, 3);
    detail::bond_idx_pairs_t<ItensorCplx> cp{{0, 1}};
    ten_t<ItensorCplx> ctr;
    trace<ItensorCplx>(ctx, cm, cp, ctr);
    CHECK(tc_test::at<ItensorCplx>(ctx, ctr, {}) == std::complex<double>(3, 0));

    destroy_context(ctx);
}

// linear_combine: with and without coefficients, real and complex
void test_linear_combine()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 1.0);
    auto b = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 2.0);
    auto c = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 3.0);

    auto r = linear_combine<ItensorReal>(ctx,
        {std::cref(a), std::cref(b), std::cref(c)}, {2.0, -1.0, 0.5});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, r, {0, 0}), 1.5));

    auto r2 = linear_combine<ItensorReal>(ctx, {std::cref(a), std::cref(b)});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, r2, {1, 1}), 3.0));

    // primed-pair (eye) inputs
    auto e1 = eye<ItensorReal>(ctx, 2);
    auto lc = linear_combine<ItensorReal>(ctx, {std::cref(e1), std::cref(e1)}, {1.0, 1.0});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, lc, {1, 1}), 2.0));

    // complex
    auto ca = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(1, 2));
    auto cb = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(3, -1));
    auto cr = linear_combine<ItensorCplx>(ctx, {std::cref(ca), std::cref(cb)},
                                          {std::complex<double>(1, 0), std::complex<double>(2, 1)});
    CHECK(tc_test::at<ItensorCplx>(ctx, cr, {0}) == std::complex<double>(1, 2) +
          std::complex<double>(2, 1) * std::complex<double>(3, -1));

    destroy_context(ctx);
}

// inverse: matrix product gives identity; singular throws (checked in errors)
void test_inverse()
{
    ItensorContext ctx; create_context(ctx);

    {
        auto M = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
        set_elem<ItensorReal>(ctx, M, {0, 0}, 4.0); set_elem<ItensorReal>(ctx, M, {0, 1}, 7.0);
        set_elem<ItensorReal>(ctx, M, {1, 0}, 2.0); set_elem<ItensorReal>(ctx, M, {1, 1}, 6.0);
        const ten_t<ItensorReal>& Mc = M;
        auto Inv = inverse<ItensorReal>(ctx, Mc, 1);
        // coordinate-wise product M * Inv = identity
        for(long i = 0; i < 2; ++i)
            for(long j = 0; j < 2; ++j)
            {
                double s = 0;
                for(long k = 0; k < 2; ++k)
                    s += tc_test::at<ItensorReal>(ctx, M, {i, k}) *
                         tc_test::at<ItensorReal>(ctx, Inv, {k, j});
                CHECK(tc_test::approx(s, (i == j) ? 1.0 : 0.0, 1e-9));
            }
    }

    // 3x3 with a known inverse
    {
        auto M = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{3, 3});
        double v[3][3] = {{2, 0, 0}, {0, 3, 1}, {0, 0, 5}};
        for(long i = 0; i < 3; ++i)
            for(long j = 0; j < 3; ++j)
                set_elem<ItensorReal>(ctx, M, {i, j}, v[i][j]);
        const ten_t<ItensorReal>& Mc = M;
        auto Inv = inverse<ItensorReal>(ctx, Mc, 1);
        for(long i = 0; i < 3; ++i)
            for(long j = 0; j < 3; ++j)
            {
                double s = 0;
                for(long k = 0; k < 3; ++k)
                    s += tc_test::at<ItensorReal>(ctx, M, {i, k}) *
                         tc_test::at<ItensorReal>(ctx, Inv, {k, j});
                CHECK(tc_test::approx(s, (i == j) ? 1.0 : 0.0, 1e-8));
            }
    }

    // complex inverse
    {
        auto M = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
        set_elem<ItensorCplx>(ctx, M, {0, 0}, std::complex<double>(1, 2));
        set_elem<ItensorCplx>(ctx, M, {0, 1}, std::complex<double>(3, -1));
        set_elem<ItensorCplx>(ctx, M, {1, 0}, std::complex<double>(0.5, 0));
        set_elem<ItensorCplx>(ctx, M, {1, 1}, std::complex<double>(-2, 1));
        const ten_t<ItensorCplx>& Mc = M;
        auto Inv = inverse<ItensorCplx>(ctx, Mc, 1);
        for(long i = 0; i < 2; ++i)
            for(long j = 0; j < 2; ++j)
            {
                std::complex<double> s = 0;
                for(long k = 0; k < 2; ++k)
                    s += tc_test::at<ItensorCplx>(ctx, M, {i, k}) *
                         tc_test::at<ItensorCplx>(ctx, Inv, {k, j});
                CHECK(tc_test::approx(s, (i == j) ? std::complex<double>(1, 0)
                                                  : std::complex<double>(0, 0), 1e-9));
            }
    }

    destroy_context(ctx);
}

// exp: identity, zero, general matrix vs scipy.expm reference, random
void test_exp()
{
    ItensorContext ctx; create_context(ctx);

    // exp of zeros = identity
    auto Z = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    const ten_t<ItensorReal>& Zc = Z;
    ten_t<ItensorReal> Ez;
    exp<ItensorReal>(ctx, Zc, 1, Ez);
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, Ez, {i, j}),
                                  (i == j) ? 1.0 : 0.0, 1e-9));

    // exp of primed-pair identity = e * I
    auto Ie = eye<ItensorReal>(ctx, 2);
    const ten_t<ItensorReal>& Iec = Ie;
    ten_t<ItensorReal> EI;
    exp<ItensorReal>(ctx, Iec, 1, EI);
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, EI, {i, j}),
                                  (i == j) ? std::exp(1.0) : 0.0, 1e-8));

    // general [[1,2],[3,4]] vs scipy.linalg.expm
    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const ten_t<ItensorReal>& Ac = A;
    ten_t<ItensorReal> E;
    exp<ItensorReal>(ctx, Ac, 1, E);
    const double ref[2][2] = {{51.968956198705, 74.73656456700321},
                              {112.10484685050484, 164.07380304920986}};
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, E, {i, j}),
                                  ref[i][j], 1e-9));

    // complex matrix vs scipy.expm
    auto Mc = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, Mc, {0, 0}, std::complex<double>(1, 2));
    set_elem<ItensorCplx>(ctx, Mc, {0, 1}, std::complex<double>(3, -1));
    set_elem<ItensorCplx>(ctx, Mc, {1, 0}, std::complex<double>(0.5, 1));
    set_elem<ItensorCplx>(ctx, Mc, {1, 1}, std::complex<double>(4, 0));
    const ten_t<ItensorCplx>& Mcc = Mc;
    ten_t<ItensorCplx> Ec;
    exp<ItensorCplx>(ctx, Mcc, 1, Ec);
    const std::complex<double> refc[2][2] = {
        {{-8.585875552001859, 20.246596659794477},
         {55.389092194887105, 38.74135487670492}},
        {{-10.790019597102356, 21.323250012045726},
         {63.96453232538542, 46.24535936570373}},
    };
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 2; ++j)
            CHECK(tc_test::approx(tc_test::at<ItensorCplx>(ctx, Ec, {i, j}),
                                  refc[i][j], 1e-8));

    // gluing property: exp(P) * exp(-P) = I
    std::mt19937 engine(1234);
    auto P = rnd(ctx, {3, 3}, engine);
    const ten_t<ItensorReal>& Pc = P;
    ten_t<ItensorReal> EP, EM;
    exp<ItensorReal>(ctx, Pc, 1, EP);
    auto Pm = P;
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 3; ++j)
            set_elem<ItensorReal>(ctx, Pm, {i, j}, -tc_test::at<ItensorReal>(ctx, Pm, {i, j}));
    const ten_t<ItensorReal>& Pmc = Pm;
    exp<ItensorReal>(ctx, Pmc, 1, EM);
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 3; ++j)
        {
            double s = 0;
            for(long k = 0; k < 3; ++k)
                s += tc_test::at<ItensorReal>(ctx, EP, {i, k}) *
                     tc_test::at<ItensorReal>(ctx, EM, {k, j});
            CHECK(tc_test::approx(s, (i == j) ? 1.0 : 0.0, 1e-8));
        }

    destroy_context(ctx);
}

// svd: reconstruction residual, orthonormal properties, real and complex
void test_svd()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(5);
    auto a = rnd(ctx, {3, 4, 12}, engine);
    ten_t<ItensorReal> u, vdag;
    real_ten_t<ItensorReal> s;
    svd<ItensorReal>(ctx, a, 2, u, s, vdag);

    // shapes
    auto su = shape<ItensorReal>(ctx, u);
    auto ss = shape<ItensorReal>(ctx, s);
    auto sv = shape<ItensorReal>(ctx, vdag);
    CHECK(su.size() == 3 && ss.size() == 2 && sv.size() == 2);
    CHECK(su[0] == 3 && su[1] == 4 && sv[1] == 12);

    // reconstruction residual: || a - u*s*vdag || (fro) <= 1e-10
    {
        double res = tc_test::svd_residual<ItensorReal>(ctx, a, u, s, vdag);
        CHECK(res < 1e-10);
    }

    // singular values: non-negative, sorted descending, squares sum to ||a||^2
    {
        auto svals = tc_test::sing_vals<ItensorReal>(ctx, s);
        for(long k = 0; k + 1 < (long)svals.size(); ++k)
        {
            CHECK(svals[k] >= 0);
            CHECK(svals[k] >= svals[k + 1]);
        }
        double s2 = 0;
        for(double x : svals) s2 += x * x;
        CHECK(tc_test::approx(s2, tc_test::frob<ItensorReal>(ctx, a) *
                                  tc_test::frob<ItensorReal>(ctx, a), 1e-8));
    }

    // complex path
    {
        std::mt19937 engine2(6);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        auto genc = [&]{ return std::complex<double>(dis(engine2), dis(engine2)); };
        auto ac = random<ItensorCplx>(ctx, shape_t<ItensorCplx>{3, 4, 12}, genc);
        ten_t<ItensorCplx> uc, vc;
        real_ten_t<ItensorCplx> sc;
        svd<ItensorCplx>(ctx, ac, 2, uc, sc, vc);
        double res = tc_test::svd_residual<ItensorCplx>(ctx, ac, uc, sc, vc);
        CHECK(res < 1e-10);
        CHECK(order<ItensorCplx>(ctx, sc) == 2);
    }

    destroy_context(ctx);
}

// trunc_svd: chi cap, truncation error, residual ordering
void test_trunc_svd()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(7);
    auto a = rnd(ctx, {3, 4, 12}, engine);
    ten_t<ItensorReal> u, vdag;
    real_ten_t<ItensorReal> s;
    real_t<ItensorReal> truncerr;
    trunc_svd<ItensorReal>(ctx, a, 2, u, s, vdag, truncerr, 2, 0.0);

    CHECK_SHAPE(ctx, s, 2, 2);
    CHECK(tc_test::at<ItensorReal>(ctx, s, {0, 1}) == 0.0);
    CHECK(truncerr >= 0.0);

    // rank-2 error <= rank-1 error
    double err2 = tc_test::svd_residual<ItensorReal>(ctx, a, u, s, vdag);

    ten_t<ItensorReal> u1, v1;
    real_ten_t<ItensorReal> s1;
    real_t<ItensorReal> e1;
    trunc_svd<ItensorReal>(ctx, a, 2, u1, s1, v1, e1, 1, 0.0);
    double err1 = tc_test::svd_residual<ItensorReal>(ctx, a, u1, s1, v1);
    CHECK(err2 < err1);
    CHECK(e1 >= 0.0);

    // complex trunc_svd keeps identical singular values/error to the full one
    {
        std::mt19937 engine2(9);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        auto genc = [&]{ return std::complex<double>(dis(engine2), dis(engine2)); };
        auto ac = random<ItensorCplx>(ctx, shape_t<ItensorCplx>{4, 4}, genc);
        ten_t<ItensorCplx> uc, vc;
        real_ten_t<ItensorCplx> sc;
        real_t<ItensorCplx> ec;
        trunc_svd<ItensorCplx>(ctx, ac, 1, uc, sc, vc, ec, 2, 0.0);
        CHECK_SHAPE(ctx, sc, 2, 2);
        double res = tc_test::svd_residual<ItensorCplx>(ctx, ac, uc, sc, vc);
        CHECK(res >= 0.0);
    }

    destroy_context(ctx);
}

// qr: shapes, reconstruction, orthonormal Q
void test_qr()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(3);
    auto a = rnd(ctx, {3, 4, 12}, engine);
    ten_t<ItensorReal> q, r;
    qr<ItensorReal>(ctx, a, 2, q, r);
    CHECK(order<ItensorReal>(ctx, q) == 3);
    CHECK(order<ItensorReal>(ctx, r) == 2);
    auto sq = shape<ItensorReal>(ctx, q);
    auto sr = shape<ItensorReal>(ctx, r);
    CHECK(sq[0] == 3 && sq[1] == 4);
    CHECK(sr[1] == 12);
    CHECK(sq[2] == sr[0]);

    // reconstruction Q * R = A
    {
        ten_t<ItensorReal> recon;
        contract<ItensorReal>(ctx, q, "ijk", r, "kl", recon, "ijl");
        double res = 0;
        for(long i = 0; i < 3; ++i)
            for(long j = 0; j < 4; ++j)
                for(long l = 0; l < 12; ++l)
                    res += std::norm(tc_test::at<ItensorReal>(ctx, recon, {i, j, l}) -
                                     tc_test::at<ItensorReal>(ctx, a, {i, j, l}));
        CHECK(std::sqrt(res) < 1e-9);
    }

    // orthonormal columns: sum_{i,j} Q(i,j,k) Q(i,j,l) = delta_{k,l}
    long link = sq[2];
    for(long k = 0; k < link; ++k)
        for(long l = 0; l < link; ++l)
        {
            double s = 0;
            for(long i = 0; i < 3; ++i)
                for(long j = 0; j < 4; ++j)
                    s += tc_test::at<ItensorReal>(ctx, q, {i, j, k}) *
                         tc_test::at<ItensorReal>(ctx, q, {i, j, l});
            CHECK(tc_test::approx(s, (k == l) ? 1.0 : 0.0, 1e-8));
        }

    // complex QR reconstruction
    {
        std::mt19937 engine2(4);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        auto genc = [&]{ return std::complex<double>(dis(engine2), dis(engine2)); };
        auto ac = random<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 3}, genc);
        ten_t<ItensorCplx> qc, rc;
        qr<ItensorCplx>(ctx, ac, 1, qc, rc);
        ten_t<ItensorCplx> recon;
        contract<ItensorCplx>(ctx, qc, "ik", rc, "kj", recon, "ij");
        double res = 0;
        for(long i = 0; i < 2; ++i)
            for(long j = 0; j < 3; ++j)
                res += std::norm(tc_test::at<ItensorCplx>(ctx, recon, {i, j}) -
                                 tc_test::at<ItensorCplx>(ctx, ac, {i, j}));
        CHECK(std::sqrt(res) < 1e-9);
    }

    destroy_context(ctx);
}

// lq: shapes, reconstruction, orthonormal Q
void test_lq()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(31);
    auto a = rnd(ctx, {3, 4, 12}, engine);
    ten_t<ItensorReal> l, q;
    lq<ItensorReal>(ctx, a, 2, l, q);
    auto sl = shape<ItensorReal>(ctx, l);
    auto sq = shape<ItensorReal>(ctx, q);
    CHECK(sl.size() == 3 && sq.size() == 2);
    CHECK(sq[1] == 12);

    // reconstruction L * Q = A
    {
        ten_t<ItensorReal> recon;
        contract<ItensorReal>(ctx, l, "ijk", q, "kl", recon, "ijl");
        for(long i = 0; i < 3; ++i)
            for(long j = 0; j < 4; ++j)
                for(long l2 = 0; l2 < 12; ++l2)
                    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, recon, {i, j, l2}),
                                          tc_test::at<ItensorReal>(ctx, a, {i, j, l2}), 1e-8));
    }

    destroy_context(ctx);
}

// eig: eigenvalue matching and the invariant A*V = V*lambda
void test_eig()
{
    ItensorContext ctx; create_context(ctx);

    // real nonsymmetric [[1,2],[3,4]] -> eigenvalues {-0.37228, 5.37228}
    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const ten_t<ItensorReal>& Ac = A;
    cplx_ten_t<ItensorReal> L, V;
    eig<ItensorReal>(ctx, Ac, 1, L, V);

    {
        auto got = tc_test::diag_elements_c(ctx, L);
        std::vector<std::complex<double>> want{{-0.3722813232690143, 0},
                                               {5.372281323269014, 0}};
        bool ok = tc_test::match_eigenvalues(got, want, 1e-9, 1e-9);
        CHECK(ok);
    }
    // invariant: A*V = V*lambda
    for(long r = 0; r < 2; ++r)
        for(long k = 0; k < 2; ++k)
        {
            std::complex<double> lhs = 0;
            for(long c = 0; c < 2; ++c)
                lhs += tc_test::at<ItensorReal>(ctx, A, {r, c}) *
                       tc_test::at<ItensorCplx>(ctx, V, {c, k});
            std::complex<double> rhs = tc_test::at<ItensorCplx>(ctx, V, {r, k}) *
                                       tc_test::diag_elements_c(ctx, L)[k];
            CHECK(tc_test::approx(lhs, rhs, 1e-9));
        }

    // rotation matrix [[0,-1],[1,0]] -> eigenvalues +/- i
    auto R = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, R, {0, 1}, -1.0);
    set_elem<ItensorReal>(ctx, R, {1, 0}, 1.0);
    const ten_t<ItensorReal>& Rc = R;
    cplx_ten_t<ItensorReal> LR, VR;
    eig<ItensorReal>(ctx, Rc, 1, LR, VR);
    {
        auto got = tc_test::diag_elements_c(ctx, LR);
        std::vector<std::complex<double>> want{{0, -1}, {0, 1}};
        bool ok = tc_test::match_eigenvalues(got, want, 1e-9, 1e-9);
        CHECK(ok);
    }

    // Hermitian complex [[2,i],[-i,3]] -> eigenvalues (5 +/- sqrt(5))/2
    auto H = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, H, {0, 0}, std::complex<double>(2, 0));
    set_elem<ItensorCplx>(ctx, H, {0, 1}, std::complex<double>(0, 1));
    set_elem<ItensorCplx>(ctx, H, {1, 0}, std::complex<double>(0, -1));
    set_elem<ItensorCplx>(ctx, H, {1, 1}, std::complex<double>(3, 0));
    const ten_t<ItensorCplx>& Hc = H;
    cplx_ten_t<ItensorCplx> LH, VH;
    eig<ItensorCplx>(ctx, Hc, 1, LH, VH);
    {
        auto got = tc_test::diag_elements_c(ctx, LH);
        std::vector<std::complex<double>> want{{1.3819660112501051, 0},
                                               {3.618033988749895, 0}};
        bool ok = tc_test::match_eigenvalues(got, want, 1e-9, 1e-9);
        CHECK(ok);
    }
    // invariant for complex eig
    for(long r = 0; r < 2; ++r)
        for(long k = 0; k < 2; ++k)
        {
            std::complex<double> lhs = 0;
            for(long c = 0; c < 2; ++c)
                lhs += tc_test::at<ItensorCplx>(ctx, H, {r, c}) *
                       tc_test::at<ItensorCplx>(ctx, VH, {c, k});
            std::complex<double> rhs = tc_test::at<ItensorCplx>(ctx, VH, {r, k}) *
                                       tc_test::diag_elements_c(ctx, LH)[k];
            CHECK(tc_test::approx(lhs, rhs, 1e-9));
        }

    destroy_context(ctx);
}

// eigvals: eigenvalues as a rank-1 tensor
void test_eigvals()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const ten_t<ItensorReal>& Ac = A;
    cplx_ten_t<ItensorReal> w;
    eigvals<ItensorReal>(ctx, Ac, 1, w);
    CHECK(order<ItensorCplx>(ctx, w) == 1);
    CHECK_SHAPE(ctx, w, 2);

    std::vector<std::complex<double>> got;
    for(long k = 0; k < 2; ++k)
        got.push_back(tc_test::at<ItensorCplx>(ctx, w, {k}));
    std::vector<std::complex<double>> want{{-0.3722813232690143, 0}, {5.372281323269014, 0}};
    CHECK(tc_test::match_eigenvalues(got, want, 1e-9, 1e-9));

    destroy_context(ctx);
}

// eigh: real symmetric and Hermitian complex, ascending eigenvalues, orthonormal V
void test_eigh()
{
    ItensorContext ctx; create_context(ctx);

    // real symmetric [[2,1],[1,2]] -> {1, 3}
    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 2.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 2.0);
    const ten_t<ItensorReal>& Ac = A;
    real_ten_t<ItensorReal> L;
    ten_t<ItensorReal> V;
    eigh<ItensorReal>(ctx, Ac, 1, L, V);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, L, {0, 0}), 1.0, 1e-9));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, L, {1, 1}), 3.0, 1e-9));

    // orthonormal columns of V
    for(long a = 0; a < 2; ++a)
        for(long b = 0; b < 2; ++b)
        {
            double d = 0;
            for(long r = 0; r < 2; ++r)
                d += tc_test::at<ItensorReal>(ctx, V, {r, a}) *
                     tc_test::at<ItensorReal>(ctx, V, {r, b});
            CHECK(tc_test::approx(d, (a == b) ? 1.0 : 0.0, 1e-9));
        }
    // reconstruction A*V = V*L
    for(long r = 0; r < 2; ++r)
        for(long k = 0; k < 2; ++k)
        {
            double s = 0;
            for(long c = 0; c < 2; ++c)
                s += tc_test::at<ItensorReal>(ctx, A, {r, c}) *
                     tc_test::at<ItensorReal>(ctx, V, {c, k});
            CHECK(tc_test::approx(s, tc_test::at<ItensorReal>(ctx, V, {r, k}) *
                                     tc_test::at<ItensorReal>(ctx, L, {k, k}), 1e-9));
        }

    // Hermitian complex [[2,i],[-i,3]] -> ascending {1.38196..., 3.61803...}
    auto H = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, H, {0, 0}, std::complex<double>(2, 0));
    set_elem<ItensorCplx>(ctx, H, {0, 1}, std::complex<double>(0, 1));
    set_elem<ItensorCplx>(ctx, H, {1, 0}, std::complex<double>(0, -1));
    set_elem<ItensorCplx>(ctx, H, {1, 1}, std::complex<double>(3, 0));
    const ten_t<ItensorCplx>& Hc = H;
    real_ten_t<ItensorCplx> LH;
    ten_t<ItensorCplx> VH;
    eigh<ItensorCplx>(ctx, Hc, 1, LH, VH);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, LH, {0, 0}), 1.3819660112501051, 1e-9));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, LH, {1, 1}), 3.618033988749895, 1e-9));

    // primed-pair (eye) eigh
    auto Ie = eye<ItensorReal>(ctx, 3);
    const ten_t<ItensorReal>& Iec = Ie;
    real_ten_t<ItensorReal> LI;
    ten_t<ItensorReal> VI;
    eigh<ItensorReal>(ctx, Iec, 1, LI, VI);
    for(long k = 0; k < 3; ++k)
        CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, LI, {k, k}), 1.0, 1e-9));

    destroy_context(ctx);
}

// eigvalsh: eigenvalues as a real rank-1 tensor
void test_eigvalsh()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 2.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 2.0);
    const ten_t<ItensorReal>& Ac = A;
    real_ten_t<ItensorReal> w;
    eigvalsh<ItensorReal>(ctx, Ac, 1, w);
    CHECK(order<ItensorReal>(ctx, w) == 1);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, w, {0}), 1.0, 1e-9));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, w, {1}), 3.0, 1e-9));

    // complex Hermitian
    auto H = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, H, {0, 0}, std::complex<double>(2, 0));
    set_elem<ItensorCplx>(ctx, H, {0, 1}, std::complex<double>(0, 1));
    set_elem<ItensorCplx>(ctx, H, {1, 0}, std::complex<double>(0, -1));
    set_elem<ItensorCplx>(ctx, H, {1, 1}, std::complex<double>(3, 0));
    const ten_t<ItensorCplx>& Hc = H;
    real_ten_t<ItensorCplx> wh;
    eigvalsh<ItensorCplx>(ctx, Hc, 1, wh);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, wh, {0}), 1.3819660112501051, 1e-9));
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, wh, {1}), 3.618033988749895, 1e-9));

    destroy_context(ctx);
}

// eig/eigh/exp are deterministic across repeated calls and never mutate inputs
void test_no_mutation_identity_preserved()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const ten_t<ItensorReal>& Ac = A;

    double snap[4];
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        snap[2 * r + c] = tc_test::at<ItensorReal>(ctx, A, {r, c});

    // eig twice
    cplx_ten_t<ItensorReal> L1, V1, L2, V2;
    eig<ItensorReal>(ctx, Ac, 1, L1, V1);
    eig<ItensorReal>(ctx, Ac, 1, L2, V2);
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            CHECK(tc_test::approx(tc_test::at<ItensorCplx>(ctx, L1, {r, k}),
                                  tc_test::at<ItensorCplx>(ctx, L2, {r, k}), 1e-12));
            CHECK(tc_test::approx(tc_test::at<ItensorCplx>(ctx, V1, {r, k}),
                                  tc_test::at<ItensorCplx>(ctx, V2, {r, k}), 1e-12));
        }
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        CHECK(snap[2 * r + c] == tc_test::at<ItensorReal>(ctx, A, {r, c}));

    // eigh twice (symmetric matrix)
    auto S = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, S, {0, 0}, 2.0); set_elem<ItensorReal>(ctx, S, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, S, {1, 0}, 1.0); set_elem<ItensorReal>(ctx, S, {1, 1}, 2.0);
    const ten_t<ItensorReal>& Sc = S;
    double snapS[4];
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        snapS[2 * r + c] = tc_test::at<ItensorReal>(ctx, S, {r, c});
    real_ten_t<ItensorReal> S1, S2;
    ten_t<ItensorReal> W1, W2;
    eigh<ItensorReal>(ctx, Sc, 1, S1, W1);
    eigh<ItensorReal>(ctx, Sc, 1, S2, W2);
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, S1, {r, k}),
                                  tc_test::at<ItensorReal>(ctx, S2, {r, k}), 1e-12));
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, W1, {r, k}),
                                  tc_test::at<ItensorReal>(ctx, W2, {r, k}), 1e-12));
        }
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        CHECK(snapS[2 * r + c] == tc_test::at<ItensorReal>(ctx, S, {r, c}));

    // exp twice (out-of-place)
    ten_t<ItensorReal> E1, E2;
    exp<ItensorReal>(ctx, Ac, 1, E1);
    exp<ItensorReal>(ctx, Ac, 1, E2);
    for(int r = 0; r < 2; ++r)
        for(int c = 0; c < 2; ++c)
        {
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, E1, {r, c}),
                                  tc_test::at<ItensorReal>(ctx, E2, {r, c}), 1e-12));
            CHECK(snap[2 * r + c] == tc_test::at<ItensorReal>(ctx, A, {r, c}));
        }

    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_norm", test_norm);
    tc_test::run_test("test_diag", test_diag);
    tc_test::run_test("test_scale", test_scale);
    tc_test::run_test("test_normalize", test_normalize);
    tc_test::run_test("test_trace", test_trace);
    tc_test::run_test("test_linear_combine", test_linear_combine);
    tc_test::run_test("test_inverse", test_inverse);
    tc_test::run_test("test_exp", test_exp);
    tc_test::run_test("test_no_mutation_identity_preserved", test_no_mutation_identity_preserved);
    tc_test::run_test("test_svd", test_svd);
    tc_test::run_test("test_trunc_svd", test_trunc_svd);
    tc_test::run_test("test_qr", test_qr);
    tc_test::run_test("test_lq", test_lq);
    tc_test::run_test("test_eig", test_eig);
    tc_test::run_test("test_eigvals", test_eigvals);
    tc_test::run_test("test_eigh", test_eigh);
    tc_test::run_test("test_eigvalsh", test_eigvalsh);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}