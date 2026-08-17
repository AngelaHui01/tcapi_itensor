#include "tc_test_util.h"
#include <complex>
#include <random>
#include <vector>

using namespace tcapi;

// allocate: shape/order, size, size_bytes, scalar (0-th order)
void test_allocate()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{3, 2, 4});
    CHECK(order<ItensorReal>(ctx, a) == 3);
    CHECK_SHAPE(ctx, a, 3, 2, 4);
    CHECK(size<ItensorReal>(ctx, a) == 24);
    CHECK(size_bytes<ItensorReal>(ctx, a) == 24 * sizeof(double));
    CHECK(tc_test::at<ItensorReal>(ctx, a, {1, 1, 1}) == 0.0);

    // 0-th order (scalar) tensor
    auto s = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{});
    CHECK(order<ItensorReal>(ctx, s) == 0);
    CHECK(size<ItensorReal>(ctx, s) == 1);

    // complex allocation
    auto ac = allocate<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    CHECK(order<ItensorCplx>(ctx, ac) == 2);
    CHECK(tc_test::at<ItensorCplx>(ctx, ac, {1, 0}) == std::complex<double>(0, 0));

    destroy_context(ctx);
}

// zeros: every element is exactly zero
void test_zeros()
{
    ItensorContext ctx; create_context(ctx);

    auto z = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{3, 4});
    double mx = tc_test::max_abs<ItensorReal>(ctx, z);
    CHECK(tc_test::approx(mx, 0.0, 1e-12));

    auto zc = zeros<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 3});
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 3; ++j)
            CHECK(tc_test::at<ItensorCplx>(ctx, zc, {i, j}) == std::complex<double>(0, 0));

    destroy_context(ctx);
}

// fill: all elements hold the given Real or Cplx value
void test_fill()
{
    ItensorContext ctx; create_context(ctx);

    auto f = fill<ItensorReal>(ctx, shape_t<ItensorReal>{3, 2, 4}, 2.5);
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 2; ++j)
            for(long k = 0; k < 4; ++k)
                CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, f, {i, j, k}), 2.5));

    auto fc = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1.0, -2.0));
    CHECK(tc_test::at<ItensorCplx>(ctx, fc, {0, 1}) == std::complex<double>(1.0, -2.0));

    destroy_context(ctx);
}

// random: elements lie in the generator's range; Real and Cplx
void test_random()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };
    auto r = random<ItensorReal>(ctx, shape_t<ItensorReal>{4, 4}, gen);
    double mx = tc_test::max_abs<ItensorReal>(ctx, r);
    CHECK(mx >= 0.0 && mx <= 1.0);

    std::mt19937 enginec(7);
    auto genc = [&]{ return std::complex<double>(dis(enginec), dis(enginec)); };
    auto rc = random<ItensorCplx>(ctx, shape_t<ItensorCplx>{3, 3}, genc);
    double a = std::abs(tc_test::at<ItensorCplx>(ctx, rc, {0, 0}));
    CHECK(a >= 0.0 && a <= std::sqrt(2.0));

    destroy_context(ctx);
}

// eye: primed-pair diagonal, 1 on the diagonal and 0 off it
void test_eye()
{
    ItensorContext ctx; create_context(ctx);

    auto e = eye<ItensorReal>(ctx, 3);
    CHECK(order<ItensorReal>(ctx, e) == 2);
    CHECK_SHAPE(ctx, e, 3, 3);
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 3; ++j)
            CHECK(tc_test::at<ItensorReal>(ctx, e, {i, j}) == (i == j ? 1.0 : 0.0));

    auto ec = eye<ItensorCplx>(ctx, 2);
    CHECK(tc_test::at<ItensorCplx>(ctx, ec, {1, 1}) == std::complex<double>(1, 0));
    CHECK(tc_test::at<ItensorCplx>(ctx, ec, {0, 1}) == std::complex<double>(0, 0));

    destroy_context(ctx);
}

// copy / move / clear
void test_copy()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{3, 4});
    set_elem<ItensorReal>(ctx, a, {0, 0}, 5.0);

    auto b = copy<ItensorReal>(ctx, a);
    CHECK(tc_test::at<ItensorReal>(ctx, b, {0, 0}) == 5.0);

    // deep copy: mutating a must not change b
    set_elem<ItensorReal>(ctx, a, {0, 0}, 9.0);
    CHECK(tc_test::at<ItensorReal>(ctx, b, {0, 0}) == 5.0);

    // move leaves the source cleared
    auto c = move<ItensorReal>(ctx, b);
    CHECK(tc_test::at<ItensorReal>(ctx, c, {0, 0}) == 5.0);
    CHECK(order<ItensorReal>(ctx, b) == 0);

    // clear resets to a 0-th order (default) tensor
    clear<ItensorReal>(ctx, c);
    CHECK(order<ItensorReal>(ctx, c) == 0);

    destroy_context(ctx);
}

// assign_from_range: row-major fill from a contiguous range
void test_assign_from_range()
{
    ItensorContext ctx; create_context(ctx);

    std::vector<double> vals{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto coors2idx = [](const elem_coors_t<ItensorReal>& coors)
    {
        return static_cast<std::size_t>(coors[0] * 3 + coors[1]);
    };
    auto a = assign_from_range<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3},
                                            vals.begin(), coors2idx);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 0}) == 1.0);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 2}) == 3.0);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {1, 2}) == 6.0);

    // complex range
    std::vector<std::complex<double>> cv{{1, 1}, {2, 0}, {-3, 4}};
    auto coors2idx2 = [](const elem_coors_t<ItensorCplx>&) { return std::size_t(0); };
    auto cc = assign_from_range<ItensorCplx>(ctx, shape_t<ItensorCplx>{3}, cv.begin(), coors2idx2);
    CHECK(tc_test::at<ItensorCplx>(ctx, cc, {0}) == std::complex<double>(1, 1));

    destroy_context(ctx);
}

// queries: order / shape / size / size_bytes / get_elem on various orders
void test_queries()
{
    ItensorContext ctx; create_context(ctx);

    auto s = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{});
    CHECK(order<ItensorReal>(ctx, s) == 0);
    CHECK(size<ItensorReal>(ctx, s) == 1);

    auto v = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{7});
    CHECK(order<ItensorReal>(ctx, v) == 1);
    CHECK(size<ItensorReal>(ctx, v) == 7);
    CHECK(size_bytes<ItensorReal>(ctx, v) == 7 * sizeof(double));

    auto m = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 5, 3});
    CHECK(order<ItensorReal>(ctx, m) == 3);
    CHECK(size<ItensorReal>(ctx, m) == 30);
    CHECK(size_bytes<ItensorReal>(ctx, m) == 30 * sizeof(double));

    // get_elem on a primed-pair tensor (eye)
    auto e = eye<ItensorReal>(ctx, 4);
    CHECK(tc_test::at<ItensorReal>(ctx, e, {2, 2}) == 1.0);
    CHECK(tc_test::at<ItensorReal>(ctx, e, {2, 3}) == 0.0);

    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_allocate", test_allocate);
    tc_test::run_test("test_zeros", test_zeros);
    tc_test::run_test("test_fill", test_fill);
    tc_test::run_test("test_random", test_random);
    tc_test::run_test("test_eye", test_eye);
    tc_test::run_test("test_copy", test_copy);
    tc_test::run_test("test_assign_from_range", test_assign_from_range);
    tc_test::run_test("test_queries", test_queries);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}