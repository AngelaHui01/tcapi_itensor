#include "tc_test_util.h"
#include <complex>
#include <cstdio>

using namespace tcapi;

// version / show / close
void test_version_show_close()
{
    ItensorContext ctx; create_context(ctx);

    CHECK(version<ItensorReal>() == "1.0");
    CHECK(version<ItensorCplx>() == "1.0");

    // show must not throw and must print something
    auto a = eye<ItensorReal>(ctx, 2);
    std::printf("-- show(eye(2)) --\n");
    show<ItensorReal>(ctx, a);

    // close: equal tensors, differing tensors
    auto b = copy<ItensorReal>(ctx, a);
    CHECK(close<ItensorReal>(ctx, a, b, 1e-12));
    CHECK(close<ItensorReal>(ctx, a, b, 0.0));

    auto c = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2}, 1.0);
    CHECK(close<ItensorReal>(ctx, a, c, 0.0) == false);
    // tolerance larger than the difference -> close
    CHECK(close<ItensorReal>(ctx, a, c, 1.5));

    // shape mismatch -> not close regardless of tolerance
    auto small = fill<ItensorReal>(ctx, shape_t<ItensorReal>{2}, 1.0);
    CHECK(close<ItensorReal>(ctx, a, small, 100.0) == false);

    // complex close comparison
    auto ca = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1, 2));
    auto cb = copy<ItensorCplx>(ctx, ca);
    CHECK(close<ItensorCplx>(ctx, ca, cb, 0.0));
    auto cc = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1, 2.5));
    CHECK(close<ItensorCplx>(ctx, ca, cc, 0.0) == false);

    destroy_context(ctx);
}

// close: negative epsilon rejected on real and complex
void test_close_negative_eps_error()
{
    ItensorContext ctx; create_context(ctx);
    auto a = eye<ItensorReal>(ctx, 2);
    auto b = copy<ItensorReal>(ctx, a);
    CHECK_THROW(std::invalid_argument, close<ItensorReal>(ctx, a, b, -0.01));

    auto ca = eye<ItensorCplx>(ctx, 2);
    auto cb = copy<ItensorCplx>(ctx, ca);
    CHECK_THROW(std::invalid_argument, close<ItensorCplx>(ctx, ca, cb, -0.01));
    destroy_context(ctx);
}

// convert: Real <-> Cplx
void test_convert()
{
    ItensorContext ctx; create_context(ctx);

    auto r = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 2});
    set_elem<ItensorReal>(ctx, r, {0, 0}, 1.5);
    set_elem<ItensorReal>(ctx, r, {1, 1}, -2.0);

    ten_t<ItensorCplx> c;
    convert<ItensorReal, ItensorCplx>(ctx, r, ctx, c);
    CHECK_SHAPE(ctx, c, 2, 2);
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {0, 0}) == std::complex<double>(1.5, 0.0));
    CHECK(tc_test::at<ItensorCplx>(ctx, c, {1, 1}) == std::complex<double>(-2.0, 0.0));

    // complex -> real drops the imaginary part
    auto h = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2}, std::complex<double>(3.0, 4.0));
    ten_t<ItensorReal> back;
    convert<ItensorCplx, ItensorReal>(ctx, h, ctx, back);
    CHECK_SHAPE(ctx, back, 2);
    CHECK(tc_test::at<ItensorReal>(ctx, back, {0}) == 3.0);

    // same-kind conversion is a deep copy
    auto r2 = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{3});
    set_elem<ItensorReal>(ctx, r2, {2}, 7.0);
    ten_t<ItensorReal> rcopy;
    convert<ItensorReal, ItensorReal>(ctx, r2, ctx, rcopy);
    CHECK(tc_test::at<ItensorReal>(ctx, rcopy, {2}) == 7.0);

    destroy_context(ctx);
}

// to_range: dump tensor into a contiguous range, recover element order
void test_to_range()
{
    ItensorContext ctx; create_context(ctx);

    auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
    for(long i = 0; i < 2; ++i)
        for(long j = 0; j < 3; ++j)
            set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 3 + j));

    std::vector<double> buf(6);
    to_range<ItensorReal>(ctx, a, buf.begin(),
        [](const elem_coors_t<ItensorReal>& coors)
        { return static_cast<std::size_t>(coors[0] * 3 + coors[1]); });
    for(std::size_t k = 0; k < buf.size(); ++k)
        CHECK(tc_test::approx(buf[k], double(k)));

    // out-of-range index from the mapping function is rejected
    CHECK_THROW(std::out_of_range,
                to_range<ItensorReal>(ctx, a, buf.begin(),
                    [](const elem_coors_t<ItensorReal>&) { return std::size_t(99); }));

    destroy_context(ctx);
}

// save / load round trip, real and complex
void test_save_load()
{
    ItensorContext ctx; create_context(ctx);

    const char* file = "tc_test_saveload.dat";

    {
        auto a = allocate<ItensorReal>(ctx, shape_t<ItensorReal>{2, 3});
        for(long i = 0; i < 2; ++i)
            for(long j = 0; j < 3; ++j)
                set_elem<ItensorReal>(ctx, a, {i, j}, double(i * 3 + j));
        save<ItensorReal>(ctx, a, file);
    }
    {
        auto b = load<ItensorReal>(ctx, std::string(file));
        CHECK_SHAPE(ctx, b, 2, 3);
        for(long i = 0; i < 2; ++i)
            for(long j = 0; j < 3; ++j)
                CHECK(tc_test::approx(tc_test::at<ItensorReal>(ctx, b, {i, j}),
                                      double(i * 3 + j)));
    }

    {
        auto c = fill<ItensorCplx>(ctx, shape_t<ItensorCplx>{2, 2}, std::complex<double>(1.5, -3.0));
        save<ItensorCplx>(ctx, c, file);
        auto d = load<ItensorCplx>(ctx, std::string(file));
        CHECK(tc_test::at<ItensorCplx>(ctx, d, {0, 0}) == std::complex<double>(1.5, -3.0));
        CHECK(tc_test::at<ItensorCplx>(ctx, d, {1, 1}) == std::complex<double>(1.5, -3.0));
    }

    std::remove(file);
    destroy_context(ctx);
}

int main()
{
    tc_test::run_test("test_version_show_close", test_version_show_close);
    tc_test::run_test("test_close_negative_eps_error", test_close_negative_eps_error);
    tc_test::run_test("test_convert", test_convert);
    tc_test::run_test("test_to_range", test_to_range);
    tc_test::run_test("test_save_load", test_save_load);

    std::printf("%d checks, %d failures\n", tc_test::g_checks, tc_test::g_failures);
    return tc_test::g_failures == 0 ? 0 : 1;
}