#include "tc_test_util.h"
#include <complex>
#include <random>
#include <vector>

using namespace tcapi;

// allocate/zeros/fill/random with {} shape
void test_constructors_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    // allocate
    auto a = allocate<ItensorReal>(ctx, {});
    CHECK(order<ItensorReal>(ctx, a) == 0);
    CHECK(size<ItensorReal>(ctx, a) == 1);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {}) == 0.0);

    // zeros
    auto z = zeros<ItensorReal>(ctx, {});
    CHECK(tc_test::at<ItensorReal>(ctx, z, {}) == 0.0);

    // fill
    auto f = fill<ItensorReal>(ctx, {}, 3.5);
    CHECK(tc_test::at<ItensorReal>(ctx, f, {}) == 3.5);

    // random
    std::mt19937 engine(42);
    auto gen = [&]{ return 2.5; };
    auto r = random<ItensorReal>(ctx, {}, gen);
    CHECK(tc_test::at<ItensorReal>(ctx, r, {}) == 2.5);

    // complex scalar
    auto cf = fill<ItensorCplx>(ctx, {}, std::complex<double>(1, 2));
    CHECK(tc_test::at<ItensorCplx>(ctx, cf, {}) == std::complex<double>(1, 2));

    destroy_context(ctx);
}

// get_elem / set_elem on zeroth-order
void test_get_set_elem_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    auto s = zeros<ItensorReal>(ctx, {});
    set_elem<ItensorReal>(ctx, s, {}, 7.0);
    CHECK(tc_test::at<ItensorReal>(ctx, s, {}) == 7.0);

    // complex
    auto cs = zeros<ItensorCplx>(ctx, {});
    set_elem<ItensorCplx>(ctx, cs, {}, std::complex<double>(3, -4));
    CHECK(tc_test::at<ItensorCplx>(ctx, cs, {}) == std::complex<double>(3, -4));

    destroy_context(ctx);
}

// assign_from_range and to_range with {}
void test_range_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    // assign_from_range
    std::vector<double> vals{42.0};
    auto a = assign_from_range<ItensorReal>(ctx, {}, vals.begin(),
        [](const elem_coors_t<ItensorReal>&) { return std::size_t(0); });
    CHECK(tc_test::at<ItensorReal>(ctx, a, {}) == 42.0);

    // to_range
    set_elem<ItensorReal>(ctx, a, {}, 99.0);
    std::vector<double> buf(1);
    to_range<ItensorReal>(ctx, a, buf.begin(),
        [](const elem_coors_t<ItensorReal>&) { return std::size_t(0); });
    CHECK(buf[0] == 99.0);

    destroy_context(ctx);
}

// for_each / for_each_with_coors on zeroth-order: callable invoked once
void test_for_each_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    {
        auto s = fill<ItensorReal>(ctx, {}, 5.0);
        long count = 0;
        for_each<ItensorReal>(ctx, s, [&](double el) { ++count; });
        CHECK(count == 1);
    }

    {
        auto s = fill<ItensorReal>(ctx, {}, 5.0);
        ten_t<ItensorReal> out;
        long count = 0;
        for_each<ItensorReal>(ctx, s, out, [&](double el) { ++count; return el + 1; });
        CHECK(count == 1);
        CHECK(tc_test::at<ItensorReal>(ctx, out, {}) == 6.0);
    }

    {
        auto s = fill<ItensorReal>(ctx, {}, 5.0);
        long count = 0;
        for_each_with_coors<ItensorReal>(ctx, s,
            [&](double el, const elem_coors_t<ItensorReal>& c) { ++count; });
        CHECK(count == 1);
    }

    // complex
    {
        auto cs = fill<ItensorCplx>(ctx, {}, std::complex<double>(2, 3));
        long count = 0;
        for_each<ItensorCplx>(ctx, cs, [&](std::complex<double> el) { ++count; });
        CHECK(count == 1);
    }

    destroy_context(ctx);
}

// reshape between {} and {1} both ways
void test_reshape_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    // scalar -> 1-element vector
    auto s = fill<ItensorReal>(ctx, {}, 7.0);
    ten_t<ItensorReal> v;
    reshape<ItensorReal>(ctx, s, {1}, v);
    CHECK_SHAPE(ctx, v, 1);
    CHECK(tc_test::at<ItensorReal>(ctx, v, {0}) == 7.0);

    // {1,1} -> scalar
    auto m = fill<ItensorReal>(ctx, {1, 1}, 3.0);
    ten_t<ItensorReal> s3;
    reshape<ItensorReal>(ctx, m, {}, s3);
    CHECK(tc_test::at<ItensorReal>(ctx, s3, {}) == 3.0);

    // 1-element vector -> scalar
    ten_t<ItensorReal> s2;
    reshape<ItensorReal>(ctx, v, {}, s2);
    CHECK(order<ItensorReal>(ctx, s2) == 0);
    CHECK(tc_test::at<ItensorReal>(ctx, s2, {}) == 7.0);

    // complex
    auto cs = fill<ItensorCplx>(ctx, {}, std::complex<double>(1, 2));
    ten_t<ItensorCplx> cv;
    reshape<ItensorCplx>(ctx, cs, {1}, cv);
    CHECK(tc_test::at<ItensorCplx>(ctx, cv, {0}) == std::complex<double>(1, 2));

    destroy_context(ctx);
}

// contract with empty output labels -> scalar
void test_contract_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, {2, 3});
    auto b = allocate<ItensorReal>(ctx, {3, 2});
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 3; ++j)
            set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 3 + j));
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 2; ++j)
            set_elem<ItensorReal>(ctx, b, {i, j}, double(i * 2 + j));

    ten_t<ItensorReal> c;
    contract<ItensorReal>(ctx, a, "ij", b, "ji", c, "");
    CHECK(order<ItensorReal>(ctx, c) == 0);

    double manual = 0.0;
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 3; ++j)
            manual += tc_test::at<ItensorReal>(ctx, a, {i, j}) *
                      tc_test::at<ItensorReal>(ctx, b, {j, i});
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, c, {}), manual, 1e-8));

    destroy_context(ctx);
}

// trace with all bonds paired -> scalar
void test_trace_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 4);
    detail::bond_idx_pairs_t<ItensorReal> full{{0, 1}};
    ten_t<ItensorReal> tr;
    trace<ItensorReal>(ctx, a, full, tr);
    CHECK(order<ItensorReal>(ctx, tr) == 0);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, tr, {}), 4.0));

    // complex trace
    auto ca = eye<ItensorCplx>(ctx, 3);
    detail::bond_idx_pairs_t<ItensorCplx> cp{{0, 1}};
    ten_t<ItensorCplx> ct;
    trace<ItensorCplx>(ctx, ca, cp, ct);
    CHECK(tc_test::at<ItensorCplx>(ctx, ct, {}) == std::complex<double>(3, 0));

    destroy_context(ctx);
}

// Bond-altering operations on zeroth-order input throw
void test_bond_ops_zero_order_throw()
{
    ItensorContext ctx; create_context(ctx);

    auto s = fill<ItensorReal>(ctx, {}, 1.0);

    // transpose requires bonds
    CHECK_THROW(std::invalid_argument,
                transpose<ItensorReal>(ctx, s, {0}));

    // expand
    Map<bond_idx_t<ItensorReal>, bond_dim_t<ItensorReal>> incmap{{0, 1}};
    CHECK_THROW(std::invalid_argument,
                expand<ItensorReal>(ctx, s, incmap));

    // concatenate with any concat_bdidx fails on zeroth-order inputs
    {
        auto t = fill<ItensorReal>(ctx, {}, 2.0);
        CHECK_THROW(std::invalid_argument,
                    concatenate<ItensorReal>(ctx, {std::cref(s), std::cref(t)}, 0));
    }

    // stack with any stack_bdidx fails on zeroth-order inputs
    {
        auto t = fill<ItensorReal>(ctx, {}, 2.0);
        CHECK_THROW(std::invalid_argument,
                    stack<ItensorReal>(ctx, {std::cref(s), std::cref(t)}, 0));
    }

    destroy_context(ctx);
}

// norm/size_bytes/close with zeroth-order
void test_queries_zero_order()
{
    ItensorContext ctx; create_context(ctx);

    auto s = fill<ItensorReal>(ctx, {}, 3.0);
    CHECK(tc_test::approx(norm<ItensorReal>(ctx, s), 3.0, 1e-12));
    CHECK(size_bytes<ItensorReal>(ctx, s) == sizeof(double));

    auto same = fill<ItensorReal>(ctx, {}, 3.0);
    CHECK(close<ItensorReal>(ctx, s, same, 0.0));

    auto diff = fill<ItensorReal>(ctx, {}, 4.0);
    CHECK(!close<ItensorReal>(ctx, s, diff, 0.0));

    // complex
    auto cs = fill<ItensorCplx>(ctx, {}, std::complex<double>(0, 1));
    CHECK(tc_test::approx(norm<ItensorCplx>(ctx, cs), 1.0, 1e-12));

    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_constructors_zero_order", test_constructors_zero_order);
    tc_test::run_test("test_get_set_elem_zero_order", test_get_set_elem_zero_order);
    tc_test::run_test("test_range_zero_order", test_range_zero_order);
    tc_test::run_test("test_for_each_zero_order", test_for_each_zero_order);
    tc_test::run_test("test_reshape_zero_order", test_reshape_zero_order);
    tc_test::run_test("test_contract_zero_order", test_contract_zero_order);
    tc_test::run_test("test_trace_zero_order", test_trace_zero_order);
    tc_test::run_test("test_bond_ops_zero_order_throw", test_bond_ops_zero_order_throw);
    tc_test::run_test("test_queries_zero_order", test_queries_zero_order);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}