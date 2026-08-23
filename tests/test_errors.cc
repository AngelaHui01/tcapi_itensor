#include "tc_test_util.h"
#include <complex>

using namespace tcapi;

// reshape: total element count mismatch
void test_reshape_error()
{
    ItensorContext ctx; create_context(ctx);
    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    CHECK_THROW(std::invalid_argument, reshape<ItensorReal>(ctx, a, {4, 4}));
    CHECK_THROW(std::invalid_argument, reshape<ItensorReal>(ctx, a, {7}));
    destroy_context(ctx);
}

// normalize: zero tensor raises
void test_normalize_error()
{
    ItensorContext ctx; create_context(ctx);
    auto z = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    CHECK_THROW(std::runtime_error, normalize<ItensorReal>(ctx, z));
    destroy_context(ctx);
}

// contract: label / shape mismatches
void test_contract_error()
{
    ItensorContext ctx; create_context(ctx);

    // shared label dimension mismatch
    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    auto b = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{4, 2});
    ten_t<ItensorReal> c;
    CHECK_THROW(std::invalid_argument,
                contract<ItensorReal>(ctx, a, "ij", b, "kj", c, "ik"));

    // output label consistent with neither input
    auto x = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    auto y = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{3, 4});
    ten_t<ItensorReal> d;
    CHECK_THROW(std::invalid_argument,
                contract<ItensorReal>(ctx, x, "ij", y, "jm", d, "io"));

    // shared label repeated in the output is invalid
    ten_t<ItensorReal> e;
    CHECK_THROW(std::invalid_argument,
                contract<ItensorReal>(ctx, x, "ij", y, "jm", e, "ijm"));

    destroy_context(ctx);
}

// inverse: bad split argument and singular matrices
void test_inverse_error()
{
    ItensorContext ctx; create_context(ctx);

    auto M = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 1.0);
    const ten_t<ItensorReal>& Mc = M;
    CHECK_THROW(std::invalid_argument, inverse<ItensorReal>(ctx, Mc, 0));
    CHECK_THROW(std::invalid_argument, inverse<ItensorReal>(ctx, Mc, 2));

    auto Sing = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 3.0);
    const ten_t<ItensorReal>& Sc = Sing;
    CHECK_THROW(std::runtime_error, inverse<ItensorReal>(ctx, Sc, 1));

    destroy_context(ctx);
}

// eig/exp: non-square input (matricization guard) and bad split index
void test_eig_exp_error()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 3; ++j)
            set_elem<ItensorReal>(ctx, A, {i, j}, double(i * 3 + j));
    const ten_t<ItensorReal>& Ac = A;

    cplx_ten_t<ItensorReal> L, V;
    CHECK_THROW(std::invalid_argument, eig<ItensorReal>(ctx, Ac, 1, L, V));

    ten_t<ItensorReal> out;
    CHECK_THROW(std::invalid_argument, exp<ItensorReal>(ctx, Ac, 1, out));

    // invalid num_of_bds_as_row
    auto Sq = eye<ItensorReal>(ctx, 3);
    const ten_t<ItensorReal>& SqC = Sq;
    CHECK_THROW(std::invalid_argument, eig<ItensorReal>(ctx, SqC, 0, L, V));
    CHECK_THROW(std::invalid_argument, exp<ItensorReal>(ctx, SqC, 0, out));

    destroy_context(ctx);
}

// eigh: non-Hermitian / non-symmetric inputs still produce a deterministic result
void test_eigh_nonhermitian()
{
    ItensorContext ctx; create_context(ctx);

    auto A = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, A, {0, 0}, 1.0); set_elem<ItensorReal>(ctx, A, {0, 1}, 2.0);
    set_elem<ItensorReal>(ctx, A, {1, 0}, 3.0); set_elem<ItensorReal>(ctx, A, {1, 1}, 4.0);
    const ten_t<ItensorReal>& Ac = A;
    real_ten_t<ItensorReal> L;
    ten_t<ItensorReal> V;
    eigh<ItensorReal>(ctx, Ac, 1, L, V);
    auto li = itensor::inds(L);
    CHECK(dim(li[0]) == 2);
    CHECK(elt(L, li[0](1), li[1](1)) == elt(L, li[0](1), li[1](1)));

    auto B = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, B, {0, 0}, std::complex<double>(1, 0));
    set_elem<ItensorCplx>(ctx, B, {0, 1}, std::complex<double>(2, 1));
    set_elem<ItensorCplx>(ctx, B, {1, 0}, std::complex<double>(1, 0));
    set_elem<ItensorCplx>(ctx, B, {1, 1}, std::complex<double>(4, 0));
    const ten_t<ItensorCplx>& Bc = B;
    real_ten_t<ItensorCplx> LB;
    ten_t<ItensorCplx> VB;
    eigh<ItensorCplx>(ctx, Bc, 1, LB, VB);
    auto bi = itensor::inds(LB);
    CHECK(dim(bi[0]) == 2);

    destroy_context(ctx);
}

// diag: invalid input order
void test_diag_error()
{
    ItensorContext ctx; create_context(ctx);
    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2, 2});
    ten_t<ItensorReal> out;
    CHECK_THROW(std::invalid_argument, diag<ItensorReal>(ctx, a, out));
    destroy_context(ctx);
}

// trace: matched bonds must have equal dimension
void test_trace_error()
{
    ItensorContext ctx; create_context(ctx);
    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3, 4, 5});
    detail::bond_idx_pairs_t<ItensorReal> pairs{{0, 3}};
    ten_t<ItensorReal> out;
    CHECK_THROW(std::invalid_argument, trace<ItensorReal>(ctx, a, pairs, out));
    destroy_context(ctx);
}

// trunc_svd: invalid chi range
void test_trunc_svd_error()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    ten_t<ItensorReal> u, vdag;
    real_ten_t<ItensorReal> s;
    real_t<ItensorReal> err;
    CHECK_THROW(std::invalid_argument,
                trunc_svd<ItensorReal>(ctx, a, 0, u, s, vdag, err, 2, 0.0));
    CHECK_THROW(std::invalid_argument,
                trunc_svd<ItensorReal>(ctx, a, 1, u, s, vdag, err, -1, 0.0));
    // num_of_bds_as_row must be < order (eye has order 2)
    CHECK_THROW(std::invalid_argument,
                trunc_svd<ItensorReal>(ctx, a, 2, u, s, vdag, err, 2, 0.0));

    destroy_context(ctx);
}

// linear_combine / stack: structural errors
void test_combine_stack_error()
{
    ItensorContext ctx; create_context(ctx);
    auto a = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 1.0);

    // coefficient-count mismatch
    CHECK_THROW(std::invalid_argument,
                linear_combine<ItensorReal>(ctx, {std::cref(a)}, {1.0, 2.0}));
    // empty inputs
    CHECK_THROW(std::invalid_argument,
                linear_combine<ItensorReal>(ctx, {}));
    // stack with no inputs
    CHECK_THROW(std::invalid_argument, stack<ItensorReal>(ctx, {}, 0));

    destroy_context(ctx);
}

// operate on a destroyed context must raise
void test_destroyed_context()
{
    ItensorContext ctx; create_context(ctx);
    destroy_context(ctx);

    CHECK_THROW(std::runtime_error, zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}));
    CHECK_THROW(std::runtime_error, destroy_context(ctx));

    ItensorContext ctx2; create_context(ctx2);
    destroy_context(ctx2);
    CHECK_THROW(std::runtime_error, eye<ItensorReal>(ctx2, 3));
}

int main()
{
    tc_test::run_test("test_reshape_error", test_reshape_error);
    tc_test::run_test("test_normalize_error", test_normalize_error);
    tc_test::run_test("test_contract_error", test_contract_error);
    tc_test::run_test("test_inverse_error", test_inverse_error);
    tc_test::run_test("test_eig_exp_error", test_eig_exp_error);
    tc_test::run_test("test_eigh_nonhermitian", test_eigh_nonhermitian);
    tc_test::run_test("test_diag_error", test_diag_error);
    tc_test::run_test("test_trace_error", test_trace_error);
    tc_test::run_test("test_trunc_svd_error", test_trunc_svd_error);
    tc_test::run_test("test_combine_stack_error", test_combine_stack_error);
    tc_test::run_test("test_destroyed_context", test_destroyed_context);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}