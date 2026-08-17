#include "tc_test_util.h"
#include <complex>
#include <random>

using namespace tcapi;

// helper: build a random tensor with the given shape using unprimed indices
static ten_t<ItensorReal> rnd(ItensorContext& ctx,
                              const shape_t<ItensorReal>& shape,
                              std::mt19937& engine)
{
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };
    return random<ItensorReal>(ctx, shape, gen);
}

// plain product (no labels) matching shared ITensor indices
void test_contract_no_labels()
{
    ItensorContext ctx; create_context(ctx);

    itensor::Index i(2, "i"), j(3, "j"), k(4, "k");
    auto a = itensor::randomITensor(i, j);
    auto b = itensor::randomITensor(j, k);
    auto c = contract<ItensorReal>(ctx, a, b);
    CHECK(order<ItensorReal>(ctx, c) == 2);
    CHECK_SHAPE(ctx, c, 2, 4);

    // scalar result: contract over i
    itensor::Index x(2, "x");
    auto v = itensor::randomITensor(x);
    auto w = itensor::randomITensor(x);
    auto s = contract<ItensorReal>(ctx, v, w);
    CHECK(order<ItensorReal>(ctx, s) == 0);

    destroy_context(ctx);
}

// labeled contract, string form (einstein convention, output may reorder)
void test_contract_labels_string()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(11);
    auto a = rnd(ctx, {2, 3, 4}, engine);  // labels i j k
    auto b = rnd(ctx, {4, 5}, engine);      // labels k l
    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ijk", b, "kl", c, "ijl");
    CHECK_SHAPE(ctx, c, 2, 3, 5);

    double manual = 0.0;
    for(long k = 0; k < 4; ++k)
        manual += tc_test::at<ItensorReal>(ctx, a, {0, 0, k}) *
                  tc_test::at<ItensorReal>(ctx, b, {k, 0});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {0, 0, 0}), manual, 1e-10));

    destroy_context(ctx);
}

// labeled contract, integer form (same semantics as string)
void test_contract_labels_int()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(13);
    auto a = rnd(ctx, {2, 3, 4}, engine);
    auto b = rnd(ctx, {4, 5}, engine);
    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, {1001, 1002, 1003},
                          b, {1003, 1004},
                          c, {1001, 1002, 1004});
    CHECK_SHAPE(ctx, c, 2, 3, 5);

    double manual = 0.0;
    for(long k = 0; k < 4; ++k)
        manual += tc_test::at<ItensorReal>(ctx, a, {0, 1, k}) *
                  tc_test::at<ItensorReal>(ctx, b, {k, 1});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {0, 1, 1}), manual, 1e-10));

    destroy_context(ctx);
}

// string and integer labels produce identical tensors
void test_contract_labels_equivalence()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(17);
    auto a = rnd(ctx, {3, 4}, engine);
    auto b = rnd(ctx, {4, 3}, engine);

    ten_t<ItensorReal> via_str;
    contract<ItensorReal>(ctx, a, "ik", b, "kj", via_str, "ij");

    ten_t<ItensorReal> via_int;
    contract<ItensorReal>(ctx, a, {7, 8}, b, {8, 9}, via_int, {7, 9});

    CHECK_ALL_CLOSE(ctx, ItensorReal, via_str, via_int, 0.0, 1e-12);

    destroy_context(ctx);
}

// summed (none-of-the-output) bonds must be contracted; repeated output label
// positions are also allowed (contract returns them in label order)
void test_contract_sum_over_common()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(19);
    auto a = rnd(ctx, {2, 3}, engine);
    auto b = rnd(ctx, {3, 4}, engine);
    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ik", b, "kj", c, "ij");
    CHECK_SHAPE(ctx, c, 2, 4);

    // sanitised positional contraction
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 4; ++j)
        {
            double manual = 0.0;
            for(long k = 0; k < 3; ++k)
                manual += tc_test::at<ItensorReal>(ctx, a, {i, k}) *
                          tc_test::at<ItensorReal>(ctx, b, {k, j});
            CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {i, j}), manual, 1e-10));
        }

    destroy_context(ctx);
}

// outer product: no common bonds, output is the full product shape
void test_contract_outer_product()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(23);
    auto a = rnd(ctx, {2, 3}, engine);
    auto b = rnd(ctx, {4, 5}, engine);
    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ij", b, "kl", c, "ijkl");
    CHECK_SHAPE(ctx, c, 2, 3, 4, 5);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {0, 0, 0, 0}),
                          tc_test::at<ItensorReal>(ctx, a, {0, 0}) *
                          tc_test::at<ItensorReal>(ctx, b, {0, 0}), 1e-10));

    destroy_context(ctx);
}

// trace-like contraction: a(i,j) * b(j,i) -> scalar
void test_contract_gives_scalar()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(29);
    auto a = rnd(ctx, {3, 4}, engine);
    auto b = rnd(ctx, {4, 3}, engine);
    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ij", b, "ji", c, "");
    CHECK(order<ItensorReal>(ctx, c) == 0);

    double manual = 0.0;
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 4; ++j)
            manual += tc_test::at<ItensorReal>(ctx, a, {i, j}) *
                      tc_test::at<ItensorReal>(ctx, b, {j, i});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {}), manual, 1e-8));

    destroy_context(ctx);
}

// complex contraction
void test_contract_complex()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 3}, std::complex<double>(1, 1));
    auto b = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{3, 2}, std::complex<double>(2, -1));
    ten_t<ItensorCplx> c;
    contract<ItensorCplx>(ctx, a, "ik", b, "kj", c, "ij");
    CHECK_SHAPE(ctx, c, 2, 2);
    // element (0,0) = sum_k (1+i)(2-i) = 3 * (2+1 + i(2-1)) = 3*(3+i)
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {0, 0}) == std::complex<double>(9, 3));

    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_contract_no_labels", test_contract_no_labels);
    tc_test::run_test("test_contract_labels_string", test_contract_labels_string);
    tc_test::run_test("test_contract_labels_int", test_contract_labels_int);
    tc_test::run_test("test_contract_labels_equivalence", test_contract_labels_equivalence);
    tc_test::run_test("test_contract_sum_over_common", test_contract_sum_over_common);
    tc_test::run_test("test_contract_outer_product", test_contract_outer_product);
    tc_test::run_test("test_contract_gives_scalar", test_contract_gives_scalar);
    tc_test::run_test("test_contract_complex", test_contract_complex);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}