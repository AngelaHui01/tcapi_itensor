// test_tcapi_itensor.cpp
#include "tcapi/tcapi.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

using namespace tcapi;

// -----------------------------------------------------------------------------
// Small helper: approximate equality for doubles
// -----------------------------------------------------------------------------
bool approx(double a, double b, double eps = 1e-10)
{
    return std::abs(a - b) < eps;
}

// -----------------------------------------------------------------------------
// Helpers for the eig/eigh/exp tests
// -----------------------------------------------------------------------------
bool approx_c(std::complex<double> a, std::complex<double> b, double eps = 1e-9)
{
    return std::abs(a - b) < eps;
}

// Real part of element (i, j), works for Real and Cplx element types.
template<typename T>
double re_of(ItensorContext& ctx, const tent_t<T>& t, int i, int j)
{
    return std::real(get_elem<T>(ctx, t, {i, j}));
}

// Imaginary part of element (i, j), 0 for Real element types.
template<typename T>
double im_of(ItensorContext& ctx, const tent_t<T>& t, int i, int j)
{
    return std::imag(get_elem<T>(ctx, t, {i, j}));
}

// -----------------------------------------------------------------------------
// Test: context lifecycle
// -----------------------------------------------------------------------------
void test_context()
{
    ItensorContext ctx;
    assert(!ctx.is_active());

    create_context(ctx);
    assert(ctx.is_active());

    destroy_context(ctx);
    assert(!ctx.is_active());

    bool threw = false;
    try { destroy_context(ctx); }
    catch(const std::runtime_error&) { threw = true; }
    assert(threw);

    std::cout << "test_context passed\n";
}

// -----------------------------------------------------------------------------
// Test: allocate / zeros / get_elem / set_elem
// -----------------------------------------------------------------------------
void test_allocate_zeros_getset()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, {3, 4, 2});
    assert(order<ItensorReal>(ctx, a) == 3);


    auto s = shape<ItensorReal>(ctx, a);
    assert(s[0] == 3 && s[1] == 4 && s[2] == 2);
    assert(size<ItensorReal>(ctx, a) == 24);
    assert(size_bytes<ItensorReal>(ctx, a) == 24 * sizeof(double));

    double v0 = get_elem<ItensorReal>(ctx, a, {0, 2, 1});
    assert(approx(v0, 0.0));

    set_elem<ItensorReal>(ctx, a, {2, 1, 0}, 1.0);

    double v1 = get_elem<ItensorReal>(ctx, a, {2, 1, 0});
    assert(approx(v1, 1.0));

    destroy_context(ctx);
}

// -----------------------------------------------------------------------------
// Test: fill
// -----------------------------------------------------------------------------
void test_fill()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, {3, 2, 4}, 2.0);
    double el = get_elem<ItensorReal>(ctx, a, {0, 1, 3});
    assert(approx(el, 2.0));

    destroy_context(ctx);
    std::cout << "test_fill passed\n";
}

// -----------------------------------------------------------------------------
// Test: eye
// -----------------------------------------------------------------------------
void test_eye()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    assert(approx(get_elem<ItensorReal>(ctx, a, {1, 1}), 1.0));
    assert(approx(get_elem<ItensorReal>(ctx, a, {1, 2}), 0.0));
    assert(approx(get_elem<ItensorReal>(ctx, a, {0, 0}), 1.0));

    destroy_context(ctx);
    std::cout << "test_eye passed\n";
}

// -----------------------------------------------------------------------------
// Test: random
// -----------------------------------------------------------------------------
void test_random()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {3, 4, 2}, gen);
    double el = get_elem<ItensorReal>(ctx, a, {1, 2, 0});
    assert(el >= 0.0 && el <= 1.0);

    destroy_context(ctx);
    std::cout << "test_random passed\n";
}

// -----------------------------------------------------------------------------
// Test: copy, move, clear (ownership utilities)
// -----------------------------------------------------------------------------
void test_copy_move_clear()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, {3, 4, 2});
    set_elem<ItensorReal>(ctx, a, {0, 0, 0}, 5.0);

    auto b = copy<ItensorReal>(ctx, a);
    assert(approx(get_elem<ItensorReal>(ctx, b, {0, 0, 0}), 5.0));

    // mutate a, b should be unaffected (deep copy check)
    set_elem<ItensorReal>(ctx, a, {0, 0, 0}, 9.0);
    assert(approx(get_elem<ItensorReal>(ctx, b, {0, 0, 0}), 5.0));

    auto acpy = copy<ItensorReal>(ctx, a);
    auto c = move<ItensorReal>(ctx, a);
    assert(approx(get_elem<ItensorReal>(ctx, c, {0, 0, 0}), 9.0));

    clear<ItensorReal>(ctx, c);
    assert(order<ItensorReal>(ctx, c) == 0); // default-constructed ITensor has order 0

    destroy_context(ctx);
    std::cout << "test_copy_move_clear passed\n";
}

// -----------------------------------------------------------------------------
// Test: assign_from_range
// -----------------------------------------------------------------------------
void test_assign_from_range()
{
    ItensorContext ctx; create_context(ctx);

    std::vector<double> vals{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto coors2idx = [](const elem_coors_t<ItensorReal>& coors)
    {
        return static_cast<std::size_t>(3 * coors[0] + coors[1]);
    };

    auto a = assign_from_range<ItensorReal>(ctx, {2, 3}, vals.begin(), coors2idx);
    double el = get_elem<ItensorReal>(ctx, a, {1, 1});
    assert(approx(el, 5.0));

    destroy_context(ctx);
    std::cout << "test_assign_from_range passed\n";
}

// -----------------------------------------------------------------------------
// Test: reshape
// -----------------------------------------------------------------------------
void test_reshape()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, {3, 4, 2});   // <-- add this back
    reshape<ItensorReal>(ctx, a, {4, 2, 3});
    auto s = shape<ItensorReal>(ctx, a);
    assert(s[0] == 4 && s[1] == 2 && s[2] == 3);

    tent_t<ItensorReal> b;
    auto c = zeros<ItensorReal>(ctx, {3, 4, 2});
    reshape<ItensorReal>(ctx, c, {2, 3, 4}, b);
    auto sb = shape<ItensorReal>(ctx, b);
    assert(sb[0] == 2 && sb[1] == 3 && sb[2] == 4);

    destroy_context(ctx);
    std::cout << "test_reshape passed\n";
}


// -----------------------------------------------------------------------------
// Test: transpose
// -----------------------------------------------------------------------------
void test_transpose()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(1);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {3, 2, 4}, gen);
    double el1 = get_elem<ItensorReal>(ctx, a, {1, 0, 0});

    transpose<ItensorReal>(ctx, a, {1, 0, 2});
    double el2 = get_elem<ItensorReal>(ctx, a, {0, 1, 0});
    assert(approx(el1, el2));

    auto b_src = random<ItensorReal>(ctx, {3, 2, 4}, gen);
    tent_t<ItensorReal> b;
    transpose<ItensorReal>(ctx, b_src, {2, 1, 0}, b);
    auto sb = shape<ItensorReal>(ctx, b);
    assert(sb[0] == 4 && sb[1] == 2 && sb[2] == 3);

    destroy_context(ctx);
    std::cout << "test_transpose passed\n";
}

// -----------------------------------------------------------------------------
// Test: cplx_conj (real no-op, complex conjugation)
// -----------------------------------------------------------------------------
void test_cplx_conj()
{
    ItensorContext ctx; create_context(ctx);

    // Real case: should be a no-op / deep copy
    auto a = fill<ItensorReal>(ctx, {2, 2}, 3.0);
    cplx_conj<ItensorReal>(ctx, a);
    assert(approx(get_elem<ItensorReal>(ctx, a, {0, 0}), 3.0));

    // Complex case
    auto c = fill<ItensorCplx>(ctx, {2, 2}, std::complex<double>(1.0, 2.0));
    tent_t<ItensorCplx> cconj;
    cplx_conj<ItensorCplx>(ctx, c, cconj);
    auto v = get_elem<ItensorCplx>(ctx, cconj, {0, 0});
    assert(approx(v.real(), 1.0) && approx(v.imag(), -2.0));

    destroy_context(ctx);
    std::cout << "test_cplx_conj passed\n";
}

// -----------------------------------------------------------------------------
// Test: for_each / for_each_with_coors
// -----------------------------------------------------------------------------
void test_for_each()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    double total = 0.0;

    // (2) read-only traversal: f(const elem)
    auto sum_elem = [&](double el){ total += el; };
    for_each<ItensorReal>(ctx, a, sum_elem);
    assert(approx(total, 3.0)); // trace of identity

    // (3) out-of-place convenience: f returns the transformed element
    auto plus_one = [](double el){ return el + 1.0; };
    tent_t<ItensorReal> c;
    for_each<ItensorReal>(ctx, a, c, plus_one);
    assert(approx(get_elem<ItensorReal>(ctx, c, {1, 1}), 2.0));
    assert(approx(get_elem<ItensorReal>(ctx, c, {0, 1}), 1.0));

    // (1) in-place: f(elem&) mutates a copy that is written back
    auto add_one_inplace = [](double& el){ el += 1.0; };
    for_each<ItensorReal>(ctx, a, add_one_inplace);
    assert(approx(get_elem<ItensorReal>(ctx, a, {0, 0}), 2.0));
    assert(approx(get_elem<ItensorReal>(ctx, a, {0, 1}), 1.0));

    destroy_context(ctx);
    std::cout << "test_for_each passed\n";
}

// -----------------------------------------------------------------------------
// Test: concatenate
// -----------------------------------------------------------------------------
void test_concatenate()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(7);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {2, 3, 4}, gen);
    auto b = random<ItensorReal>(ctx, {2, 3, 4}, gen);
    auto c = random<ItensorReal>(ctx, {2, 3, 4}, gen);

    auto d = concatenate<ItensorReal>(ctx, {a, b, c}, 1);
    auto sd = shape<ItensorReal>(ctx, d);
    assert(sd[0] == 2 && sd[1] == 9 && sd[2] == 4);

    double el1 = get_elem<ItensorReal>(ctx, b, {0, 0, 0});
    double el2 = get_elem<ItensorReal>(ctx, d, {0, 3, 0});
    assert(approx(el1, el2));

    destroy_context(ctx);
    std::cout << "test_concatenate passed\n";
}

// -----------------------------------------------------------------------------
// Test: norm and diag
// -----------------------------------------------------------------------------
void test_norm_diag()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    double n = norm<ItensorReal>(ctx, a);
    assert(approx(n, std::sqrt(3.0))); // Frobenius norm of I_3

    // diag on 1st-order tensor -> 2nd-order diagonal
    auto vec = fill<ItensorReal>(ctx, {3}, 2.0);
    tent_t<ItensorReal> diag_out;
    diag<ItensorReal>(ctx, vec, diag_out);
    assert(approx(get_elem<ItensorReal>(ctx, diag_out, {1, 1}), 2.0));
    assert(approx(get_elem<ItensorReal>(ctx, diag_out, {0, 1}), 0.0));

    // diag on 2nd-order tensor -> extract diagonal
    tent_t<ItensorReal> extracted;
    diag<ItensorReal>(ctx, a, extracted);
    assert(order<ItensorReal>(ctx, extracted) == 1);
    assert(approx(get_elem<ItensorReal>(ctx, extracted, {0}), 1.0));

    destroy_context(ctx);
    std::cout << "test_norm_diag passed\n";
}

// -----------------------------------------------------------------------------
// Test: contract
// -----------------------------------------------------------------------------
void test_contract()
{
    ItensorContext ctx; create_context(ctx);

    itensor::Index i(2, "i"), j(3, "j"), k(4, "k");
    auto a = itensor::randomITensor(i, j);
    auto b = itensor::randomITensor(j, k);

    auto c = contract<ItensorReal>(ctx, a, b);
    auto sc = shape<ItensorReal>(ctx, c);
    assert(sc.size() == 2); // result has bonds i, k after summing over j

    destroy_context(ctx);
    std::cout << "test_contract passed\n";
}

// -----------------------------------------------------------------------------
// Test: qr
// -----------------------------------------------------------------------------
void test_qr()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(3);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {3, 4, 12}, gen);
    tent_t<ItensorReal> q, r;
    qr<ItensorReal>(ctx, a, 2, q, r);

    // shape sanity: q keeps first 2 bonds + new bond; r has new bond + remaining bond
    auto sq = shape<ItensorReal>(ctx, q);
    auto sr = shape<ItensorReal>(ctx, r);
    assert(sq.size() == 3);
    assert(sr.size() == 2);

    destroy_context(ctx);
    std::cout << "test_qr passed\n";
}

// -----------------------------------------------------------------------------
// Small helper: Frobenius norm of (a - b) for real or complex tensors
// -----------------------------------------------------------------------------
template<typename TenT>
double frob_err(const context_handle_t<TenT>& ctx,
                const tent_t<TenT>& a, const tent_t<TenT>& b)
{
    auto dims = shape<TenT>(ctx, a);
    double err = 0.0;
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        auto d = get_elem<TenT>(ctx, a, coors) - get_elem<TenT>(ctx, b, coors);
        err += std::norm(d);
    });
    return std::sqrt(err);
}

// -----------------------------------------------------------------------------
// Test: svd
// -----------------------------------------------------------------------------
void test_svd()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(5);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {3, 4, 12}, gen);
    tent_t<ItensorReal> u, vdag;
    real_ten_t<ItensorReal> s;
    svd<ItensorReal>(ctx, a, 2, u, s, vdag);

    // sigma should be diagonal, order 2
    assert(order<ItensorReal>(ctx, s) == 2);

    // shape sanity: u=(3,4,kappa), s=(kappa,kappa), vdag=(kappa,12)
    auto su = shape<ItensorReal>(ctx, u);
    auto ss = shape<ItensorReal>(ctx, s);
    auto sv = shape<ItensorReal>(ctx, vdag);
    assert(su.size() == 3 && ss.size() == 2 && sv.size() == 2);
    assert(su[0] == 3 && su[1] == 4 && sv[1] == 12);
    assert(su[2] == sv[0] && su[2] == ss[0] && ss[0] == ss[1]);

    // reconstruction: u * s * vdag == a
    auto recon = contract<ItensorReal>(ctx, contract<ItensorReal>(ctx, u, s), vdag);
    assert(frob_err<ItensorReal>(ctx, a, recon) < 1e-10);

    // complex path: conj(V) would break reconstruction
    std::mt19937 engineC(6);
    auto genC = [&]{ return std::complex<double>(dis(engineC), dis(engineC)); };
    auto ac = random<ItensorCplx>(ctx, {3, 4, 12}, genC);
    tent_t<ItensorCplx> uc, vdagc;
    real_ten_t<ItensorCplx> sc;
    svd<ItensorCplx>(ctx, ac, 2, uc, sc, vdagc);
    auto reconc = contract<ItensorCplx>(ctx, contract<ItensorCplx>(ctx, uc, sc), vdagc);
    assert(frob_err<ItensorCplx>(ctx, ac, reconc) < 1e-10);

    destroy_context(ctx);
    std::cout << "test_svd passed\n";
}

// -----------------------------------------------------------------------------
// Test: trunc_svd
// -----------------------------------------------------------------------------
void test_trunc_svd()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(7);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {3, 4, 12}, gen);
    tent_t<ItensorReal> u, vdag;
    real_ten_t<ItensorReal> s;
    real_t<ItensorReal> truncerr;
    trunc_svd<ItensorReal>(ctx, a, 2, u, s, vdag, truncerr, 2, 0.0);

    auto su = shape<ItensorReal>(ctx, u);
    auto ss = shape<ItensorReal>(ctx, s);
    auto sv = shape<ItensorReal>(ctx, vdag);
    assert(su[2] == 2 && ss[0] == 2 && ss[1] == 2 && sv[0] == 2); // MaxDim cap
    assert(truncerr >= 0.0);

    // truncated reconstruction should match the best rank-2 approximation:
    // its error must be <= the error of the best rank-1 approximation
    auto recon = contract<ItensorReal>(ctx, contract<ItensorReal>(ctx, u, s), vdag);
    auto err2 = frob_err<ItensorReal>(ctx, a, recon);

    tent_t<ItensorReal> u1, v1;
    real_ten_t<ItensorReal> s1;
    real_t<ItensorReal> e1;
    trunc_svd<ItensorReal>(ctx, a, 2, u1, s1, v1, e1, 1, 0.0);
    auto recon1 = contract<ItensorReal>(ctx, contract<ItensorReal>(ctx, u1, s1), v1);
    auto err1 = frob_err<ItensorReal>(ctx, a, recon1);
    assert(err2 < err1);

    destroy_context(ctx);
    std::cout << "test_trunc_svd passed\n";
}

// -----------------------------------------------------------------------------
// Test: scale / normalize
// -----------------------------------------------------------------------------
void test_scale_normalize()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, {2, 2}, 2.0);
    scale<ItensorReal>(ctx, a, 3.0);
    assert(approx(get_elem<ItensorReal>(ctx, a, {0, 0}), 6.0));

    tent_t<ItensorReal> b;
    auto src = fill<ItensorReal>(ctx, {2, 2}, 2.0);
    scale<ItensorReal>(ctx, src, 5.0, b);
    assert(approx(get_elem<ItensorReal>(ctx, b, {1, 1}), 10.0));
    assert(approx(get_elem<ItensorReal>(ctx, src, {1, 1}), 2.0)); // src unmodified

    auto c = fill<ItensorReal>(ctx, {2, 2}, 3.0);
    double n = normalize<ItensorReal>(ctx, c);
    assert(approx(n, std::sqrt(4 * 9.0)));
    assert(approx(norm<ItensorReal>(ctx, c), 1.0));

    destroy_context(ctx);
    std::cout << "test_scale_normalize passed\n";
}

// -----------------------------------------------------------------------------
// Test: contract with explicit bond labels
// -----------------------------------------------------------------------------
void test_contract_labels()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(11);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {2, 3, 4}, gen);   // labels i,j,k
    auto b = random<ItensorReal>(ctx, {4, 5}, gen);       // labels k,l

    tent_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ijk", b, "kl", c, "ijl");
    auto sc = shape<ItensorReal>(ctx, c);
    assert(sc[0] == 2 && sc[1] == 3 && sc[2] == 5);

    // Manual check against a brute-force sum for one output element.
    double manual = 0.0;
    for(long k = 0; k < 4; ++k)
        manual += get_elem<ItensorReal>(ctx, a, {0, 0, k}) *
                  get_elem<ItensorReal>(ctx, b, {k, 0});
    assert(approx(get_elem<ItensorReal>(ctx, c, {0, 0, 0}), manual));

    destroy_context(ctx);
    std::cout << "test_contract_labels passed\n";
}

// -----------------------------------------------------------------------------
// Test: exp (matrix exponential on a Hermitian 2-index tensor)
// -----------------------------------------------------------------------------
void test_exp()
{
    ItensorContext ctx; create_context(ctx);

    auto i = itensor::Index(3, "b0");
    itensor::ITensor z(i, itensor::prime(i));
    // fill with a small Hermitian perturbation instead of all zeros
    for(int r = 1; r <= 3; ++r)
        for(int c = 1; c <= 3; ++c)
            z.set(i(r), itensor::prime(i)(c), (r == c) ? 0.0 : 0.0);
    // still zero, but let's actually test exp(0) differently:
    // just check exp(0) == identity via a *non-square-degenerate* sanity check instead

    exp<ItensorReal>(ctx, z, 1);
    assert(approx(get_elem<ItensorReal>(ctx, z, {0, 0}), 1.0));
    assert(approx(get_elem<ItensorReal>(ctx, z, {0, 1}), 0.0));

    destroy_context(ctx);
    std::cout << "test_exp passed\n";
}

// -----------------------------------------------------------------------------
// Test: eigh (Hermitian eigensolver on identity)
// -----------------------------------------------------------------------------
void test_eigh()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    real_ten_t<ItensorReal> lambda_mat;
    tent_t<ItensorReal> v;
    eigh<ItensorReal>(ctx, a, 1, lambda_mat, v);

    // All eigenvalues of I_3 should be 1.
    assert(order<ItensorReal>(ctx, lambda_mat) == 2);
    assert(approx(get_elem<ItensorReal>(ctx, lambda_mat, {0, 0}), 1.0));
    assert(approx(get_elem<ItensorReal>(ctx, lambda_mat, {1, 1}), 1.0));
    assert(approx(get_elem<ItensorReal>(ctx, lambda_mat, {2, 2}), 1.0));

    destroy_context(ctx);
    std::cout << "test_eigh passed\n";
}

// -----------------------------------------------------------------------------
// Test: inverse (in-place and out-of-place, Real and Cplx)
// -----------------------------------------------------------------------------
template<typename T>
static void check_inverse_identity(ItensorContext& ctx, double r00, double i00,
                                   double r01, double i01, double r10, double i10,
                                   double r11, double i11)
{
    auto v = [](double re, double im) {
        if constexpr(std::is_same<T, ItensorCplx>::value)
            return std::complex<double>(re, im);
        else
            return re;
    };

    auto M = allocate<T>(ctx, {2, 2});
    set_elem<T>(ctx, M, {0, 0}, v(r00, i00));
    set_elem<T>(ctx, M, {0, 1}, v(r01, i01));
    set_elem<T>(ctx, M, {1, 0}, v(r10, i10));
    set_elem<T>(ctx, M, {1, 1}, v(r11, i11));

    const tent_t<T>& Mc = M;
    auto InvU = allocate<T>(ctx, {2, 2});
    inverse<T>(ctx, Mc, 1, InvU);
    auto Inv = inverse<T>(ctx, Mc, 1);

    double Mi[2][2], Ui[2][2], Ii[2][2];
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
        {
            Mi[i][j] = std::real(get_elem<T>(ctx, M, {i, j}));
            Ui[i][j] = std::real(get_elem<T>(ctx, InvU, {i, j}));
            Ii[i][j] = std::real(get_elem<T>(ctx, Inv, {i, j}));
        }
    for(int i = 0; i < 2; ++i) assert(approx(Ui[i][0], Ii[i][0]) && approx(Ui[i][1], Ii[i][1]));

    double P[2][2] = {};
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            for(int k = 0; k < 2; ++k)
                P[i][k] += Mi[i][j] * Ii[j][k];
    for(int i = 0; i < 2; ++i)
        for(int k = 0; k < 2; ++k)
            assert(approx(P[i][k], (i == k) ? 1.0 : 0.0));
}

void test_inverse()
{
    ItensorContext ctx; create_context(ctx);

    check_inverse_identity<ItensorReal>(ctx, 4, 0, 7, 0, 2, 0, 6, 0);
    check_inverse_identity<ItensorCplx>(ctx, 1, 2, 3, -1, 0.5, 0, -2, 1);

    // Error paths: 2x2 tensor needs num_of_bds_as_row == 1.
    {
        auto M = fill<ItensorReal>(ctx, {2, 2}, 1.0);
        const tent_t<ItensorReal>& Mc = M;
        for(int k : {0, 2})
        {
            bool threw = false;
            try { inverse<ItensorReal>(ctx, Mc, k); }
            catch(const std::invalid_argument&) { threw = true; }
            assert(threw);
        }
    }

    // Singular matrix must throw.
    {
        auto S = fill<ItensorReal>(ctx, {2, 2}, 3.0);
        const tent_t<ItensorReal>& Sc = S;
        bool threw = false;
        try { inverse<ItensorReal>(ctx, Sc, 1); }
        catch(const std::runtime_error&) { threw = true; }
        assert(threw);
    }

    destroy_context(ctx);
    std::cout << "test_inverse passed\n";
}

// -----------------------------------------------------------------------------
// Test: real / imag / to_cplx
// -----------------------------------------------------------------------------
void test_real_imag_to_cplx()
{
    ItensorContext ctx; create_context(ctx);

    auto c = fill<ItensorCplx>(ctx, {2, 2}, std::complex<double>(3.0, -4.0));

    auto r = real<ItensorCplx>(ctx, c);
    assert(approx(get_elem<ItensorCplx>(ctx, c, {0, 0}).real(), 3.0));
    assert(approx(itensor::elt(r, itensor::inds(r)[0](1), itensor::inds(r)[1](1)), 3.0));

    auto im = imag<ItensorCplx>(ctx, c);
    assert(approx(itensor::elt(im, itensor::inds(im)[0](1), itensor::inds(im)[1](1)), -4.0));

    // real -> complex round trip
    auto rr = fill<ItensorReal>(ctx, {2, 2}, 5.0);
    auto cc = to_cplx<ItensorReal>(ctx, rr);
    auto v = get_elem<ItensorCplx>(ctx, cc, {0, 0});
    assert(approx(v.real(), 5.0) && approx(v.imag(), 0.0));

    destroy_context(ctx);
    std::cout << "test_real_imag_to_cplx passed\n";
}

// -----------------------------------------------------------------------------
// Test: expand
// -----------------------------------------------------------------------------
void test_expand()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, {2, 3}, 7.0);

    tent_t<ItensorReal> out;
    Map<bond_idx_t<ItensorReal>, bond_dim_t<ItensorReal>> incmap{{1, 2}};
    expand<ItensorReal>(ctx, a, incmap, out);

    auto s = shape<ItensorReal>(ctx, out);
    assert(s[0] == 2 && s[1] == 5);
    assert(approx(get_elem<ItensorReal>(ctx, out, {0, 2}), 7.0));   // original region
    assert(approx(get_elem<ItensorReal>(ctx, out, {0, 4}), 0.0));   // appended region

    // in-place overload
    auto b = fill<ItensorReal>(ctx, {2, 3}, 1.0);
    expand<ItensorReal>(ctx, b, incmap);
    auto sb = shape<ItensorReal>(ctx, b);
    assert(sb[1] == 5);

    destroy_context(ctx);
    std::cout << "test_expand passed\n";
}

// -----------------------------------------------------------------------------
// Test: shrink
// -----------------------------------------------------------------------------
void test_shrink()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(13);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {4, 5}, gen);
    double ref = get_elem<ItensorReal>(ctx, a, {1, 2});

    tent_t<ItensorReal> out;
    detail::bond_idx_elem_coor_pair_map_t<ItensorReal> ranges{{1, {1, 4}}};
    shrink<ItensorReal>(ctx, a, ranges, out);

    auto s = shape<ItensorReal>(ctx, out);
    assert(s[0] == 4 && s[1] == 3);
    assert(approx(get_elem<ItensorReal>(ctx, out, {1, 1}), ref)); // (1,2) -> (1,1)

    destroy_context(ctx);
    std::cout << "test_shrink passed\n";
}

// -----------------------------------------------------------------------------
// Test: extract_sub
// -----------------------------------------------------------------------------
void test_extract_sub()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(17);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, {5, 5}, gen);
    double ref = get_elem<ItensorReal>(ctx, a, {2, 3});

    List<Pair<elem_coor_t<ItensorReal>, elem_coor_t<ItensorReal>>> coor_pairs{{1, 4}, {2, 5}};
    tent_t<ItensorReal> out;
    extract_sub<ItensorReal>(ctx, a, coor_pairs, out);

    auto s = shape<ItensorReal>(ctx, out);
    assert(s[0] == 3 && s[1] == 3);
    assert(approx(get_elem<ItensorReal>(ctx, out, {1, 1}), ref)); // (2,3) -> (1,1)

    destroy_context(ctx);
    std::cout << "test_extract_sub passed\n";
}

// -----------------------------------------------------------------------------
// Test: replace_sub
// -----------------------------------------------------------------------------
void test_replace_sub()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, {4, 4});
    auto sub = fill<ItensorReal>(ctx, {2, 2}, 9.0);

    tent_t<ItensorReal> out;
    replace_sub<ItensorReal>(ctx, a, sub, {1, 1}, out);

    assert(approx(get_elem<ItensorReal>(ctx, out, {1, 1}), 9.0));
    assert(approx(get_elem<ItensorReal>(ctx, out, {2, 2}), 9.0));
    assert(approx(get_elem<ItensorReal>(ctx, out, {0, 0}), 0.0));
    assert(approx(get_elem<ItensorReal>(ctx, out, {3, 3}), 0.0));

    destroy_context(ctx);
    std::cout << "test_replace_sub passed\n";
}

// -----------------------------------------------------------------------------
// Test: trace (partial trace, avoiding the rank-0 edge case)
// -----------------------------------------------------------------------------
void test_trace()
{
    ItensorContext ctx; create_context(ctx);

    // Build a 4th-order tensor as an outer product of two 3x3 identities,
    // then trace out one matched pair -> should yield 3 * I_3.
    auto id = eye<ItensorReal>(ctx, 3);

    shape_t<ItensorReal> full_shape{3, 3, 3, 3};
    auto t = allocate<ItensorReal>(ctx, full_shape);
    auto is_t = itensor::inds(t);
    detail::for_each_coordinate<ItensorReal>(full_shape, [&](const elem_coors_t<ItensorReal>& c)
    {
        double v = get_elem<ItensorReal>(ctx, id, {c[0], c[1]}) *
                   get_elem<ItensorReal>(ctx, id, {c[2], c[3]});
        detail::set_elem_impl<ItensorReal>(t, detail::to_ivs<ItensorReal>(is_t, c), v);
    });

    // Trace over bonds (2,3): sum_k id[2,3]-block delta_{k,k} -> factor 3, keep bonds 0,1.
    detail::bond_idx_pairs_t<ItensorReal> pairs{{2, 3}};
    tent_t<ItensorReal> out;
    trace<ItensorReal>(ctx, t, pairs, out);

    auto s = shape<ItensorReal>(ctx, out);
    assert(s.size() == 2 && s[0] == 3 && s[1] == 3);
    assert(approx(get_elem<ItensorReal>(ctx, out, {0, 0}), 3.0));
    assert(approx(get_elem<ItensorReal>(ctx, out, {0, 1}), 0.0));

    destroy_context(ctx);
    std::cout << "test_trace passed\n";
}

// -----------------------------------------------------------------------------
// Test: linear_combine
// -----------------------------------------------------------------------------
void test_linear_combine()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, {2, 2}, 1.0);
    auto b = fill<ItensorReal>(ctx, {2, 2}, 2.0);
    auto c = fill<ItensorReal>(ctx, {2, 2}, 3.0);

    auto r = linear_combine<ItensorReal>(ctx, {a, b, c}, {2.0, -1.0, 0.5});
    // 2*1 + (-1)*2 + 0.5*3 = 2 - 2 + 1.5 = 1.5
    assert(approx(get_elem<ItensorReal>(ctx, r, {0, 0}), 1.5));

    auto r2 = linear_combine<ItensorReal>(ctx, {a, b});
    // default coefs = 1 -> 1 + 2 = 3
    assert(approx(get_elem<ItensorReal>(ctx, r2, {1, 1}), 3.0));

    destroy_context(ctx);
    std::cout << "test_linear_combine passed\n";
}

// -----------------------------------------------------------------------------
// Test: stack
// -----------------------------------------------------------------------------
void test_stack()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, {2, 3}, 1.0);
    auto b = fill<ItensorReal>(ctx, {2, 3}, 2.0);
    auto c = fill<ItensorReal>(ctx, {2, 3}, 3.0);

    // Stack at bond index 0 -> new order-3 tensor of shape (3, 2, 3)
    auto s = stack<ItensorReal>(ctx, {a, b, c}, 0);
    auto sh = shape<ItensorReal>(ctx, s);
    assert(sh.size() == 3 && sh[0] == 3 && sh[1] == 2 && sh[2] == 3);

    assert(approx(get_elem<ItensorReal>(ctx, s, {0, 0, 0}), 1.0));
    assert(approx(get_elem<ItensorReal>(ctx, s, {1, 0, 0}), 2.0));
    assert(approx(get_elem<ItensorReal>(ctx, s, {2, 1, 2}), 3.0));

    // Stack at trailing bond index (== base order) -> shape (2, 3, 3)
    auto s2 = stack<ItensorReal>(ctx, {a, b, c}, 2);
    auto sh2 = shape<ItensorReal>(ctx, s2);
    assert(sh2.size() == 3 && sh2[0] == 2 && sh2[1] == 3 && sh2[2] == 3);
    assert(approx(get_elem<ItensorReal>(ctx, s2, {0, 0, 1}), 2.0));

    destroy_context(ctx);
    std::cout << "test_stack passed\n";
}
// -----------------------------------------------------------------------------
// Test: exp on general (possibly nonsymmetric) matrices, vs numpy expm
// -----------------------------------------------------------------------------
void test_exp_general()
{
    ItensorContext ctx; create_context(ctx);

    // unprimed matrix from allocate: exp([[1,2],[3,4]]) vs numpy expm
    auto A = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const tent_t<ItensorReal>& Ac = A;
    tent_t<ItensorReal> E;
    exp<ItensorReal>(ctx, Ac, 1, E);
    const double ref[2][2] = {{51.968956198705, 74.73656456700321},
                              {112.10484685050484, 164.07380304920986}};
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            assert(approx(get_elem<ItensorReal>(ctx, E, {i, j}), ref[i][j], 1e-9));

    // unprimed zeros: exp(0) must be the identity
    auto Z = zeros<ItensorReal>(ctx, {2, 2});
    const tent_t<ItensorReal>& Zc = Z;
    tent_t<ItensorReal> Ez;
    exp<ItensorReal>(ctx, Zc, 1, Ez);
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            assert(approx(get_elem<ItensorReal>(ctx, Ez, {i, j}), (i == j) ? 1.0 : 0.0, 1e-9));

    // primed matrix from eye: exp(I) = e*I
    auto Ie = eye<ItensorReal>(ctx, 2);
    const tent_t<ItensorReal>& Iec = Ie;
    tent_t<ItensorReal> EI;
    exp<ItensorReal>(ctx, Iec, 1, EI);
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            assert(approx(get_elem<ItensorReal>(ctx, EI, {i, j}),
                          (i == j) ? std::exp(1.0) : 0.0, 1e-8));

    // matrix whose eigenvalues are a complex-conjugate pair (+-i): the
    // general (non-Hermitian) path must produce exp(rotation) = rotation(1)
    auto R = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, R, {0, 0}, 0.0);    set_elem<ItensorReal>(ctx, R, {0, 1}, -1.0);
    set_elem<ItensorReal>(ctx, R, {1, 0}, 1.0);    set_elem<ItensorReal>(ctx, R, {1, 1}, 0.0);
    const tent_t<ItensorReal>& Rc = R;
    tent_t<ItensorReal> Er;
    exp<ItensorReal>(ctx, Rc, 1, Er);
    const double c1 = std::cos(1.0), s1 = std::sin(1.0);
    const double rot[2][2] = {{c1, -s1}, {s1, c1}};
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            assert(approx(get_elem<ItensorReal>(ctx, Er, {i, j}), rot[i][j], 1e-9));

    // complex matrix vs numpy expm
    auto Mc = allocate<ItensorCplx>(ctx, {2, 2});
    set_elem<ItensorCplx>(ctx, Mc, {0, 0}, std::complex<double>(1, 2));
    set_elem<ItensorCplx>(ctx, Mc, {0, 1}, std::complex<double>(3, -1));
    set_elem<ItensorCplx>(ctx, Mc, {1, 0}, std::complex<double>(0.5, 1));
    set_elem<ItensorCplx>(ctx, Mc, {1, 1}, std::complex<double>(4, 0));
    const tent_t<ItensorCplx>& Mcc = Mc;
    tent_t<ItensorCplx> Ec;
    exp<ItensorCplx>(ctx, Mcc, 1, Ec);
    const std::complex<double> refc[2][2] = {
        {{-8.585875552001859, 20.246596659794477},
         {55.389092194887105, 38.74135487670492}},
        {{-10.790019597102356, 21.323250012045726},
         {63.96453232538542, 46.24535936570373}},
    };
    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
        {
            auto v = get_elem<ItensorCplx>(ctx, Ec, {i, j});
            assert(approx_c(v, refc[i][j], 1e-8));
        }

    destroy_context(ctx);
    std::cout << "test_exp_general passed\n";
}

// -----------------------------------------------------------------------------
// Test: exp on a random unprimed matrix satisfies exp(P)*exp(-P) = I and is
// repeatable (no mutation of the source).
// -----------------------------------------------------------------------------
void test_exp_random_property()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 rng(1234);
    auto unif = std::uniform_real_distribution<double>(-1.0, 1.0);
    auto gen = [&]() { return unif(rng); };
    auto P = random<ItensorReal>(ctx, {3, 3}, gen);
    const tent_t<ItensorReal>& Pc = P;

    tent_t<ItensorReal> EP, EM;
    exp<ItensorReal>(ctx, Pc, 1, EP);
    // exp(-P): flip sign in a copy
    auto Pm = P;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            set_elem<ItensorReal>(ctx, Pm, {r, c},
                                  -get_elem<ItensorReal>(ctx, Pm, {r, c}));
    const tent_t<ItensorReal>& Pmc = Pm;
    exp<ItensorReal>(ctx, Pmc, 1, EM);

    // positional matrix product EP * EM must be the identity
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
        {
            double s = 0;
            for(int k = 0; k < 3; ++k)
                s += get_elem<ItensorReal>(ctx, EP, {r, k}) * get_elem<ItensorReal>(ctx, EM, {k, c});
            assert(approx(s, (r == c) ? 1.0 : 0.0, 1e-8));
        }

    // repeatability + no source mutation
    double snap[9];
    for(int r = 0; r < 3; ++r) for(int c = 0; c < 3; ++c)
        snap[3 * r + c] = get_elem<ItensorReal>(ctx, P, {r, c});
    tent_t<ItensorReal> EP2;
    exp<ItensorReal>(ctx, Pc, 1, EP2);
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
        {
            assert(approx(get_elem<ItensorReal>(ctx, EP, {r, c}),
                          get_elem<ItensorReal>(ctx, EP2, {r, c}), 1e-12));
            assert(approx(snap[3 * r + c], get_elem<ItensorReal>(ctx, P, {r, c}), 0.0));
        }

    destroy_context(ctx);
    std::cout << "test_exp_random_property passed\n";
}

// -----------------------------------------------------------------------------
// Test: eig on general square matrices (real nonsymmetric, complex-conjugate
// eigenvalue pairs, complex input). Reconstruction A*V = V*Lambda is checked
// positionally, mirroring the numpy backend test convention.
// -----------------------------------------------------------------------------
void test_eig_general()
{
    ItensorContext ctx; create_context(ctx);

    // nonsymmetric real matrix [[1,2],[3,4]]: eigenvalues {-0.37228, 5.37228}
    auto A = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const tent_t<ItensorReal>& Ac = A;
    cplx_ten_t<ItensorReal> L, V;
    eig<ItensorReal>(ctx, Ac, 1, L, V);
    double w[2], wn[2] = {-0.3722813232690143, 5.372281323269014};
    for(int k = 0; k < 2; ++k) w[k] = re_of<ItensorReal>(ctx, L, k, k);
    std::sort(w, w + 2); std::sort(wn, wn + 2);
    for(int k = 0; k < 2; ++k) assert(approx(w[k], wn[k], 1e-9));
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            std::complex<double> s = 0;
            for(int c = 0; c < 2; ++c)
                s += get_elem<ItensorReal>(ctx, A, {r, c}) * get_elem<ItensorCplx>(ctx, V, {c, k});
            std::complex<double> rhs = get_elem<ItensorCplx>(ctx, V, {r, k}) *
                                       get_elem<ItensorCplx>(ctx, L, {k, k});
            assert(approx_c(s, rhs, 1e-9));
        }

    // real matrix with a complex-conjugate eigenvalue pair: rotation (+-i)
    auto R = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, R, {0, 1}, -1.0);
    set_elem<ItensorReal>(ctx, R, {1, 0}, 1.0);
    const tent_t<ItensorReal>& Rc = R;
    cplx_ten_t<ItensorReal> LR, VR;
    eig<ItensorReal>(ctx, Rc, 1, LR, VR);
    double realw[2] = {re_of<ItensorReal>(ctx, LR, 0, 0), re_of<ItensorReal>(ctx, LR, 1, 1)};
    double imagw[2] = {im_of<ItensorReal>(ctx, LR, 0, 0), im_of<ItensorReal>(ctx, LR, 1, 1)};
    for(int k = 0; k < 2; ++k)
        assert(approx(std::abs(realw[k]), 0.0, 1e-9) && approx(std::abs(imagw[k]), 1.0, 1e-9));

    // complex-Hermitian input [[2,i],[-i,3]]: eigenvalues {(5+-sqrt(5))/2}
    auto H = allocate<ItensorCplx>(ctx, {2, 2});
    set_elem<ItensorCplx>(ctx, H, {0, 0}, std::complex<double>(2, 0));
    set_elem<ItensorCplx>(ctx, H, {0, 1}, std::complex<double>(0, 1));
    set_elem<ItensorCplx>(ctx, H, {1, 0}, std::complex<double>(0, -1));
    set_elem<ItensorCplx>(ctx, H, {1, 1}, std::complex<double>(3, 0));
    const tent_t<ItensorCplx>& Hc = H;
    cplx_ten_t<ItensorCplx> LH, VH;
    eig<ItensorCplx>(ctx, Hc, 1, LH, VH);
    double wh[2] = {re_of<ItensorCplx>(ctx, LH, 0, 0), re_of<ItensorCplx>(ctx, LH, 1, 1)};
    double whn[2] = {1.3819660112501051, 3.618033988749895};
    std::sort(wh, wh + 2); std::sort(whn, whn + 2);
    for(int k = 0; k < 2; ++k) assert(approx(wh[k], whn[k], 1e-9));
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            std::complex<double> s = 0;
            for(int c = 0; c < 2; ++c)
                s += get_elem<ItensorCplx>(ctx, H, {r, c}) * get_elem<ItensorCplx>(ctx, VH, {c, k});
            std::complex<double> rhs = get_elem<ItensorCplx>(ctx, VH, {r, k}) *
                                       get_elem<ItensorCplx>(ctx, LH, {k, k});
            assert(approx_c(s, rhs, 1e-9));
        }

    destroy_context(ctx);
    std::cout << "test_eig_general passed\n";
}

// -----------------------------------------------------------------------------
// Test: eigh on real symmetric and complex-Hermitian matrices: ascending real
// eigenvalues, unitary eigenvectors, A*V = V*Lambda reconstruction.
// -----------------------------------------------------------------------------
void test_eigh_symmetric()
{
    ItensorContext ctx; create_context(ctx);

    // real symmetric from allocate (unprimed): [[2,1],[1,2]] -> {1,3}
    auto A = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 2.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 2.0);
    const tent_t<ItensorReal>& Ac = A;
    real_ten_t<ItensorReal> L;
    tent_t<ItensorReal> V;
    eigh<ItensorReal>(ctx, Ac, 1, L, V);
    assert(approx(re_of<ItensorReal>(ctx, L, 0, 0), 1.0, 1e-9));
    assert(approx(re_of<ItensorReal>(ctx, L, 1, 1), 3.0, 1e-9));
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            double s = 0;
            for(int c = 0; c < 2; ++c)
                s += get_elem<ItensorReal>(ctx, A, {r, c}) * get_elem<ItensorReal>(ctx, V, {c, k});
            assert(approx(s, get_elem<ItensorReal>(ctx, V, {r, k}) *
                             get_elem<ItensorReal>(ctx, L, {k, k}), 1e-9));
        }
    // orthonormal columns of V
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
        {
            double d = 0;
            for(int r = 0; r < 2; ++r)
                d += get_elem<ItensorReal>(ctx, V, {r, a}) * get_elem<ItensorReal>(ctx, V, {r, b});
            assert(approx(d, (a == b) ? 1.0 : 0.0, 1e-9));
        }

    // complex-Hermitian from allocate: [[2,i],[-i,3]] -> ascending real evals
    auto H = allocate<ItensorCplx>(ctx, {2, 2});
    set_elem<ItensorCplx>(ctx, H, {0, 0}, std::complex<double>(2, 0));
    set_elem<ItensorCplx>(ctx, H, {0, 1}, std::complex<double>(0, 1));
    set_elem<ItensorCplx>(ctx, H, {1, 0}, std::complex<double>(0, -1));
    set_elem<ItensorCplx>(ctx, H, {1, 1}, std::complex<double>(3, 0));
    const tent_t<ItensorCplx>& Hc = H;
    real_ten_t<ItensorCplx> LH;
    tent_t<ItensorCplx> VH;
    eigh<ItensorCplx>(ctx, Hc, 1, LH, VH);
    assert(approx(re_of<ItensorCplx>(ctx, LH, 0, 0), 1.3819660112501051, 1e-9));
    assert(approx(re_of<ItensorCplx>(ctx, LH, 1, 1), 3.618033988749895, 1e-9));
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            std::complex<double> s = 0;
            for(int c = 0; c < 2; ++c)
                s += get_elem<ItensorCplx>(ctx, H, {r, c}) * get_elem<ItensorCplx>(ctx, VH, {c, k});
            std::complex<double> rhs = get_elem<ItensorCplx>(ctx, VH, {r, k}) *
                                       get_elem<ItensorCplx>(ctx, LH, {k, k});
            assert(approx_c(s, rhs, 1e-9));
        }

    // primed matrix from eye: eigh(I) = {1,1}
    auto Ie = eye<ItensorReal>(ctx, 2);
    const tent_t<ItensorReal>& Iec = Ie;
    real_ten_t<ItensorReal> LI;
    tent_t<ItensorReal> VI;
    eigh<ItensorReal>(ctx, Iec, 1, LI, VI);
    assert(approx(re_of<ItensorReal>(ctx, LI, 0, 0), 1.0, 1e-9));
    assert(approx(re_of<ItensorReal>(ctx, LI, 1, 1), 1.0, 1e-9));
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            double s = 0;
            for(int c = 0; c < 2; ++c)
                s += get_elem<ItensorReal>(ctx, Ie, {r, c}) * get_elem<ItensorReal>(ctx, VI, {c, k});
            assert(approx(s, get_elem<ItensorReal>(ctx, VI, {r, k}), 1e-9));
        }

    destroy_context(ctx);
    std::cout << "test_eigh_symmetric passed\n";
}

// -----------------------------------------------------------------------------
// Test: eigh rejects non-Hermitian input with a clean std::invalid_argument
// -----------------------------------------------------------------------------
void test_eigh_rejects_nonsymmetric()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const tent_t<ItensorReal>& Ac = A;
    real_ten_t<ItensorReal> L;
    tent_t<ItensorReal> V;
    bool threw = false;
    try { eigh<ItensorReal>(ctx, Ac, 1, L, V); }
    catch(const std::invalid_argument&) { threw = true; }
    assert(threw);

    // nonsymmetric complex input must also be rejected
    auto B = allocate<ItensorCplx>(ctx, {2, 2});
    set_elem<ItensorCplx>(ctx, B, {0, 0}, std::complex<double>(1, 0));
    set_elem<ItensorCplx>(ctx, B, {0, 1}, std::complex<double>(2, 1));
    set_elem<ItensorCplx>(ctx, B, {1, 0}, std::complex<double>(1, 0));
    set_elem<ItensorCplx>(ctx, B, {1, 1}, std::complex<double>(4, 0));
    const tent_t<ItensorCplx>& Bc = B;
    real_ten_t<ItensorCplx> LB;
    tent_t<ItensorCplx> VB;
    threw = false;
    try { eigh<ItensorCplx>(ctx, Bc, 1, LB, VB); }
    catch(const std::invalid_argument&) { threw = true; }
    assert(threw);

    destroy_context(ctx);
    std::cout << "test_eigh_rejects_nonsymmetric passed\n";
}

// -----------------------------------------------------------------------------
// Test: repeated calls leave the source tensor untouched and give identical
// results (no mutation, no index leakage) for eig, eigh and exp.
// -----------------------------------------------------------------------------
void test_no_mutation_identity_preserved()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const tent_t<ItensorReal>& Ac = A;

    double snap[4];
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        snap[2 * r + c] = get_elem<ItensorReal>(ctx, A, {r, c});

    // eig twice
    cplx_ten_t<ItensorReal> L1, V1, L2, V2;
    eig<ItensorReal>(ctx, Ac, 1, L1, V1);
    eig<ItensorReal>(ctx, Ac, 1, L2, V2);
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            assert(approx_c(get_elem<ItensorCplx>(ctx, L1, {r, k}), get_elem<ItensorCplx>(ctx, L2, {r, k}), 1e-12));
            assert(approx_c(get_elem<ItensorCplx>(ctx, V1, {r, k}), get_elem<ItensorCplx>(ctx, V2, {r, k}), 1e-12));
        }
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        assert(approx(snap[2 * r + c], get_elem<ItensorReal>(ctx, A, {r, c}), 0.0));

    // eigh twice (symmetric matrix)
    auto S = allocate<ItensorReal>(ctx, {2, 2});
    set_elem<ItensorReal>(ctx, S, {0, 0}, 2.0); set_elem<ItensorReal>(ctx, S, {0, 1}, 1.0);
    set_elem<ItensorReal>(ctx, S, {1, 0}, 1.0); set_elem<ItensorReal>(ctx, S, {1, 1}, 2.0);
    const tent_t<ItensorReal>& Sc = S;
    double snapS[4];
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        snapS[2 * r + c] = get_elem<ItensorReal>(ctx, S, {r, c});
    real_ten_t<ItensorReal> S1, S2;
    tent_t<ItensorReal> W1, W2;
    eigh<ItensorReal>(ctx, Sc, 1, S1, W1);
    eigh<ItensorReal>(ctx, Sc, 1, S2, W2);
    for(int r = 0; r < 2; ++r)
        for(int k = 0; k < 2; ++k)
        {
            assert(approx(get_elem<ItensorReal>(ctx, S1, {r, k}), get_elem<ItensorReal>(ctx, S2, {r, k}), 1e-12));
            assert(approx(get_elem<ItensorReal>(ctx, W1, {r, k}), get_elem<ItensorReal>(ctx, W2, {r, k}), 1e-12));
        }
    for(int r = 0; r < 2; ++r) for(int c = 0; c < 2; ++c)
        assert(approx(snapS[2 * r + c], get_elem<ItensorReal>(ctx, S, {r, c}), 0.0));

    // exp twice (out-of-place)
    tent_t<ItensorReal> E1, E2;
    exp<ItensorReal>(ctx, Ac, 1, E1);
    exp<ItensorReal>(ctx, Ac, 1, E2);
    for(int r = 0; r < 2; ++r)
        for(int c = 0; c < 2; ++c)
        {
            assert(approx(get_elem<ItensorReal>(ctx, E1, {r, c}), get_elem<ItensorReal>(ctx, E2, {r, c}), 1e-12));
            assert(approx(snap[2 * r + c], get_elem<ItensorReal>(ctx, A, {r, c}), 0.0));
        }

    destroy_context(ctx);
    std::cout << "test_no_mutation_identity_preserved passed\n";
}

int main()
{
    test_context();
    test_allocate_zeros_getset();
    test_fill();
    test_eye();
    test_random();
    test_copy_move_clear();
    test_assign_from_range();
    test_reshape();
    test_transpose();
    test_cplx_conj();
    test_for_each();
    test_concatenate();
    test_norm_diag();
    test_contract();
    test_qr();
    test_svd();
    test_trunc_svd();

    // Section 9 additions
    test_scale_normalize();
    test_contract_labels();
    test_exp();
    test_eigh();
    test_inverse();

    // Section 11 additions
    test_real_imag_to_cplx();
    test_expand();
    test_shrink();
    test_extract_sub();
    test_replace_sub();
    test_trace();
    test_linear_combine();
    test_stack();

    // exp / eig / eigh general-matrix coverage (dense path)
    test_exp_general();
    test_exp_random_property();
    test_eig_general();
    test_eigh_symmetric();
    test_eigh_rejects_nonsymmetric();
    test_no_mutation_identity_preserved();

    std::cout << "\nAll tests passed!\n";
    return 0;
}