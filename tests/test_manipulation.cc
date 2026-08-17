#include "tc_test_util.h"
#include <complex>
#include <random>

using namespace tcapi;

// set_elem on Real and Cplx (including scalar)
void test_set_elem()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    set_elem<ItensorReal>(ctx, a, {1, 2}, 4.5);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, a, {1, 2}), 4.5));
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 0}) == 0.0);

    auto c = zeros<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2});
    set_elem<ItensorCplx>(ctx, c, {0, 1}, std::complex<double>(1.0, -2.0));
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {0, 1}) == std::complex<double>(1.0, -2.0));

    // scalar
    auto s = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{});
    set_elem<ItensorReal>(ctx, s, {}, 3.0);
    CHECK(tc_test::at<ItensorReal>(ctx, s, {}) == 3.0);

    destroy_context(ctx);
}

// transpose: real and complex, in-place and out-of-place, primed-pair source
void test_transpose()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(1);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, shape_t<ItensorReal>{3, 4, 2}, gen);
    double src = tc_test::at<ItensorReal>(ctx, a, {1, 2, 0});
    transpose<ItensorReal>(ctx, a, {1, 0, 2});
    CHECK_SHAPE(ctx, a, 4, 3, 2);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, a, {2, 1, 0}), src));

    ten_t<ItensorReal> b;
    transpose<ItensorReal>(ctx, a, {2, 1, 0}, b);
    CHECK_SHAPE(ctx, b, 2, 3, 4);

    // complex + primed-pair (eye)
    auto ec = eye<ItensorCplx>(ctx, 3);
    ten_t<ItensorCplx> et;
    transpose<ItensorCplx>(ctx, ec, {1, 0}, et);
    CHECK_SHAPE(ctx, et, 3, 3);
    CHECK(tc_test::at<ItensorCplx>(ctx, et, {1, 1}) == std::complex<double>(1, 0));

    destroy_context(ctx);
}

// reshape: real and complex, in-place and out-of-place
void test_reshape()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{3, 4});
    for(long i = 0; i < 3; ++i)
        for(long j = 0; j < 4; ++j)
            set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 4 + j));
    reshape<ItensorReal>(ctx, a, {6, 2});
    CHECK_SHAPE(ctx, a, 6, 2);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, a, {1, 1}), 3.0)); // flat idx 3

    ten_t<ItensorReal> b;
    reshape<ItensorReal>(ctx, a, {12}, b);
    CHECK_SHAPE(ctx, b, 12);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, b, {11}), 11.0));

    // complex
    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 6}, std::complex<double>(1, 2));
    reshape<ItensorCplx>(ctx, c, {4, 3});
    CHECK_SHAPE(ctx, c, 4, 3);

    // primed-pair source (eye) reshaped
    auto e = eye<ItensorReal>(ctx, 2);
    ten_t<ItensorReal> er;
    reshape<ItensorReal>(ctx, e, {4}, er);
    CHECK_SHAPE(ctx, er, 4);

    destroy_context(ctx);
}

// cplx_conj: no-op for Real, conjugation for Cplx
void test_cplx_conj()
{
    ItensorContext ctx; create_context(ctx);

    auto r = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 3.0);
    cplx_conj<ItensorReal>(ctx, r);
    CHECK(tc_test::at<ItensorReal>(ctx, r, {0, 0}) == 3.0);

    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1.0, 2.0));
    ten_t<ItensorCplx> cc;
    cplx_conj<ItensorCplx>(ctx, c, cc);
    CHECK(tc_test::at<ItensorCplx>(ctx, cc, {0, 0}) == std::complex<double>(1.0, -2.0));
    // input unchanged
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {0, 0}) == std::complex<double>(1.0, 2.0));

    destroy_context(ctx);
}

// for_each variants
void test_for_each()
{
    ItensorContext ctx; create_context(ctx);

    auto a = eye<ItensorReal>(ctx, 3);
    double total = 0.0;
    auto sum = [&](double el) { total += el; };
    for_each<ItensorReal>(ctx, a, sum);
    CHECK(tc_test::approx(total, 3.0));

    // out-of-place transform
    ten_t<ItensorReal> b;
    auto plus_one = [](double el) { return el + 1.0; };
    for_each<ItensorReal>(ctx, a, b, plus_one);
    CHECK(tc_test::at<ItensorReal>(ctx, b, {1, 1}) == 2.0);
    CHECK(tc_test::at<ItensorReal>(ctx, b, {0, 1}) == 1.0);

    // in-place
    auto add_one = [](double& el) { el += 1.0; };
    for_each<ItensorReal>(ctx, a, add_one);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 0}) == 2.0);

    // complex in-place
    auto cc = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(1, 1));
    auto conj = [](std::complex<double>& el) { el = std::conj(el); };
    for_each<ItensorCplx>(ctx, cc, conj);
    CHECK(tc_test::at<ItensorCplx>(ctx, cc, {0}) == std::complex<double>(1, -1));

    destroy_context(ctx);
}

// for_each_with_coors
void test_for_each_with_coors()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    auto fill_fn = [](double& el, const elem_coors_t<ItensorReal>& coors)
    {
        el = double(coors[0] * 100 + coors[1]);
    };
    for_each_with_coors<ItensorReal>(ctx, a, fill_fn);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {0, 0}) == 0.0);
    CHECK(tc_test::at<ItensorReal>(ctx, a, {1, 2}) == 102.0);

    // out-of-place with coors on complex
    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(0, 0));
    ten_t<ItensorCplx> out;
    auto scale_by = [](std::complex<double> el, const elem_coors_t<ItensorCplx>& coors)
    {
        return std::complex<double>(double(coors[0] + coors[1]), 0);
    };
    for_each_with_coors<ItensorCplx>(ctx, c, out, scale_by);
    CHECK(tc_test::at<ItensorCplx>(ctx, out, {1, 1}) == std::complex<double>(2, 0));

    destroy_context(ctx);
}

// concatenate: real and complex
void test_concatenate()
{
    ItensorContext ctx; create_context(ctx);

    std::mt19937 engine(7);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto gen = [&]{ return dis(engine); };

    auto a = random<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3, 4}, gen);
    auto b = random<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3, 4}, gen);
    auto c = random<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3, 4}, gen);
    auto d = concatenate<ItensorReal>(ctx, {std::cref(a), std::cref(b), std::cref(c)}, 1);
    CHECK_SHAPE(ctx, d, 2, 9, 4);
    CHECK(tc_test::at<ItensorReal>(ctx, d, {0, 0, 0}) == tc_test::at<ItensorReal>(ctx, a, {0, 0, 0}));
    CHECK(tc_test::at<ItensorReal>(ctx, d, {0, 3, 0}) == tc_test::at<ItensorReal>(ctx, b, {0, 0, 0}));
    CHECK(tc_test::at<ItensorReal>(ctx, d, {0, 6, 0}) == tc_test::at<ItensorReal>(ctx, c, {0, 0, 0}));

    // complex concatenation
    auto ca = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1, 1));
    auto cb = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 3}, std::complex<double>(2, 2));
    auto cd = concatenate<ItensorCplx>(ctx, {std::cref(ca), std::cref(cb)}, 1);
    CHECK_SHAPE(ctx, cd, 2, 5);
    CHECK(tc_test::at<ItensorCplx>(ctx, cd, {1, 4}) == std::complex<double>(2, 2));

    destroy_context(ctx);
}

// stack: real and complex
void test_stack()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3}, 1.0);
    auto b = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3}, 2.0);
    auto c = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3}, 3.0);

    auto s = stack<ItensorReal>(ctx, {std::cref(a), std::cref(b), std::cref(c)}, 0);
    CHECK_SHAPE(ctx, s, 3, 2, 3);
    CHECK(tc_test::at<ItensorReal>(ctx, s, {0, 0, 0}) == 1.0);
    CHECK(tc_test::at<ItensorReal>(ctx, s, {2, 1, 2}) == 3.0);

    auto ca = eye<ItensorCplx>(ctx, 2);
    auto cb = eye<ItensorCplx>(ctx, 2);
    auto cs = stack<ItensorCplx>(ctx, {std::cref(ca), std::cref(cb)}, 0);
    CHECK_SHAPE(ctx, cs, 2, 2, 2);
    CHECK(tc_test::at<ItensorCplx>(ctx, cs, {1, 1, 1}) == std::complex<double>(1, 0));

    destroy_context(ctx);
}

// expand
void test_expand()
{
    ItensorContext ctx; create_context(ctx);

    auto a = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3}, 7.0);
    ten_t<ItensorReal> out;
    Map<bond_idx_t<ItensorReal>, bond_dim_t<ItensorReal>> incmap{{1, 2}};
    expand<ItensorReal>(ctx, a, incmap, out);
    CHECK_SHAPE(ctx, out, 2, 5);
    CHECK(tc_test::at<ItensorReal>(ctx, out, {0, 2}) == 7.0);
    CHECK(tc_test::at<ItensorReal>(ctx, out, {0, 4}) == 0.0);

    // complex + primed-pair
    auto ec = eye<ItensorCplx>(ctx, 2);
    expand<ItensorCplx>(ctx, ec, incmap);
    CHECK_SHAPE(ctx, ec, 2, 4);
    CHECK(tc_test::at<ItensorCplx>(ctx, ec, {1, 1}) == std::complex<double>(1, 0));
    CHECK(tc_test::at<ItensorCplx>(ctx, ec, {1, 2}) == std::complex<double>(0, 0));

    destroy_context(ctx);
}

// shrink
void test_shrink()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{4, 5});
    for(long i = 0; i < 4; ++i)
        for(long j = 0; j < 5; ++j)
            set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 10 + j));

    ten_t<ItensorReal> out;
    detail::bond_idx_elem_coor_pair_map<ItensorReal> ranges{{1, {1, 4}}};
    shrink<ItensorReal>(ctx, a, ranges, out);
    CHECK_SHAPE(ctx, out, 4, 3);
    // a(1,2) -> out(1,1)
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, out, {1, 1}), 12.0));

    auto id = eye<ItensorCplx>(ctx, 3);
    detail::bond_idx_elem_coor_pair_map<ItensorCplx> rangesc{{0, {1, 3}}};
    ten_t<ItensorCplx> outc;
    shrink<ItensorCplx>(ctx, id, rangesc, outc);
    CHECK_SHAPE(ctx, outc, 2, 3);
    // out(a,b) = eye(a+1, b); eye(1,1) = 1
    CHECK(tc_test::at<ItensorCplx>(ctx, outc, {0, 1}) == std::complex<double>(1, 0));

    destroy_context(ctx);
}

// extract_sub
void test_extract_sub()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{5, 5});
    for(long i = 0; i < 5; ++i)
        for(long j = 0; j < 5; ++j)
            set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 10 + j));

    List<Pair<elem_coor_t<ItensorReal>, elem_coor_t<ItensorReal>>> pairs{{1, 4}, {2, 5}};
    ten_t<ItensorReal> out;
    extract_sub<ItensorReal>(ctx, a, pairs, out);
    CHECK_SHAPE(ctx, out, 3, 3);
    // a(2,3) -> out(1,1)
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, out, {1, 1}), 23.0));

    auto id = eye<ItensorCplx>(ctx, 3);
    List<Pair<elem_coor_t<ItensorCplx>, elem_coor_t<ItensorCplx>>> pairsc{{0, 2}, {1, 3}};
    ten_t<ItensorCplx> outc;
    extract_sub<ItensorCplx>(ctx, id, pairsc, outc);
    CHECK_SHAPE(ctx, outc, 2, 2);
    CHECK(tc_test::at<ItensorCplx>(ctx, outc, {1, 0}) == std::complex<double>(1, 0)); // id(1,1)

    destroy_context(ctx);
}

// replace_sub
void test_replace_sub()
{
    ItensorContext ctx; create_context(ctx);

    auto a = zeros<ItensorReal>(ctx, shape_t<ItensorReal>{4, 4});
    auto sub = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 9.0);
    ten_t<ItensorReal> out;
    replace_sub<ItensorReal>(ctx, a, sub, elem_coors_t<ItensorReal>{1, 1}, out);
    CHECK_SHAPE(ctx, out, 4, 4);
    CHECK(tc_test::at<ItensorReal>(ctx, out, {1, 1}) == 9.0);
    CHECK(tc_test::at<ItensorReal>(ctx, out, {2, 2}) == 9.0);
    CHECK(tc_test::at<ItensorReal>(ctx, out, {0, 0}) == 0.0);

    // complex + primed-pair
    auto ip = eye<ItensorCplx>(ctx, 5);
    auto subc = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{3, 3}, std::complex<double>(2, 1));
    ten_t<ItensorCplx> outp;
    replace_sub<ItensorCplx>(ctx, ip, subc, elem_coors_t<ItensorCplx>{1, 1}, outp);
    // inside the replaced block
    CHECK(tc_test::at<ItensorCplx>(ctx, outp, {1, 1}) == std::complex<double>(2, 1));
    CHECK(tc_test::at<ItensorCplx>(ctx, outp, {3, 3}) == std::complex<double>(2, 1));
    // outside the replaced block the identity map is preserved
    CHECK(tc_test::at<ItensorCplx>(ctx, outp, {0, 0}) == std::complex<double>(1, 0));
    CHECK(tc_test::at<ItensorCplx>(ctx, outp, {4, 4}) == std::complex<double>(1, 0));
    CHECK(tc_test::at<ItensorCplx>(ctx, outp, {0, 1}) == std::complex<double>(0, 0));

    destroy_context(ctx);
}

// real / imag / to_cplx
void test_real_imag_to_cplx()
{
    ItensorContext ctx; create_context(ctx);

    auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(3.0, -4.0));
    auto r = real<ItensorCplx>(ctx, c);
    CHECK_SHAPE(ctx, r, 2, 2);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, r, {0, 0}), 3.0));

    auto im = imag<ItensorCplx>(ctx, c);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, im, {0, 0}), -4.0));

    auto rr = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 5.0);
    auto cc = to_cplx<ItensorReal>(ctx, rr);
    CHECK(tc_test::at<ItensorCplx>(ctx, cc, {0, 0}) == std::complex<double>(5.0, 0.0));
    // to_cplx on an already-complex tensor is a deep copy
    auto cc2 = to_cplx<ItensorCplx>(ctx, c);
    CHECK(tc_test::at<ItensorCplx>(ctx, cc2, {0, 0}) == std::complex<double>(3.0, -4.0));
    // real/imag on a Real tensor
    auto rr2 = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2}, 2.0);
    auto rim = imag<ItensorReal>(ctx, rr2);
    CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, rim, {0}), 0.0));

    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_set_elem", test_set_elem);
    tc_test::run_test("test_transpose", test_transpose);
    tc_test::run_test("test_reshape", test_reshape);
    tc_test::run_test("test_cplx_conj", test_cplx_conj);
    tc_test::run_test("test_for_each", test_for_each);
    tc_test::run_test("test_for_each_with_coors", test_for_each_with_coors);
    tc_test::run_test("test_concatenate", test_concatenate);
    tc_test::run_test("test_stack", test_stack);
    tc_test::run_test("test_expand", test_expand);
    tc_test::run_test("test_shrink", test_shrink);
    tc_test::run_test("test_extract_sub", test_extract_sub);
    tc_test::run_test("test_replace_sub", test_replace_sub);
    tc_test::run_test("test_real_imag_to_cplx", test_real_imag_to_cplx);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}