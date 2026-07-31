#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

#include <sparrow/primitive_array.hpp>

#include "sparrow-compute/arithmetic.hpp"
#include "sparrow-compute/math.hpp"
#include "sparrow-compute/operators.hpp"

namespace
{
    template <class T, std::ranges::input_range R>
    void expect_values(const sparrow::primitive_array<T>& actual, R&& expected)
    {
        ASSERT_EQ(actual.size(), static_cast<std::size_t>(std::ranges::distance(expected)));
        EXPECT_TRUE(std::ranges::equal(
            actual,
            expected,
            {},
            [](const auto& value) { return value.value(); }
        ));
    }

    template <class E, std::ranges::input_range R>
    void expect_values(const xt::xexpression<E>& expression, R&& expected)
    {
        expect_values(sparrow::compute::eval(expression), std::forward<R>(expected));
    }
}  // namespace

template <class T>
concept has_sqrt = requires(const sparrow::primitive_array<T>& a)
{
    sparrow::compute::sqrt(sparrow::compute::as_expression(a));
};

static_assert(has_sqrt<double>);
static_assert(has_sqrt<int32_t>);

// ---------------------------------------------------------------------------
// Functional kernels
// ---------------------------------------------------------------------------

TEST(Kernels, AddDoubles)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0, 4.0};
    sparrow::primitive_array<double> b{10.0, 20.0, 30.0, 40.0};

    auto r = sparrow::compute::add(a, b);
    ASSERT_EQ(r.size(), 4u);
    expect_values(r, (std::vector<double>{11.0, 22.0, 33.0, 44.0}));
}
TEST(Kernels, SubtractInts)
{
    sparrow::primitive_array<int32_t> a{10, 20, 30, 40};
    sparrow::primitive_array<int32_t> b{1, 2, 3, 4};

    auto r = sparrow::compute::subtract(a, b);
    ASSERT_EQ(r.size(), 4u);
    expect_values(r, (std::vector<int32_t>{9, 18, 27, 36}));
}

TEST(Kernels, MultiplyDoubles)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
    sparrow::primitive_array<double> b{2.0, 3.0, 4.0};

    auto r = sparrow::compute::multiply(a, b);
    ASSERT_EQ(r.size(), 3u);
    expect_values(r, (std::vector<double>{2.0, 6.0, 12.0}));
}

TEST(Kernels, DivideDoubles)
{
    sparrow::primitive_array<double> a{10.0, 20.0, 30.0};
    sparrow::primitive_array<double> b{2.0, 5.0, 10.0};

    auto r = sparrow::compute::divide(a, b);
    ASSERT_EQ(r.size(), 3u);
    expect_values(r, (std::vector<double>{5.0, 4.0, 3.0}));
}

TEST(Kernels, DivideIntsTruncates)
{
    sparrow::primitive_array<int32_t> a{7, 8, 9};
    sparrow::primitive_array<int32_t> b{2, 3, 4};

    auto r = sparrow::compute::divide(a, b);
    ASSERT_EQ(r.size(), 3u);
    expect_values(r, (std::vector<int32_t>{3, 2, 2}));
}

TEST(Operators, ArrayAndScalarArithmetic)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
    sparrow::primitive_array<double> b{4.0, 5.0, 6.0};

    expect_values(a + b, (std::vector<double>{5.0, 7.0, 9.0}));
    expect_values(a - b, (std::vector<double>{-3.0, -3.0, -3.0}));
    expect_values(a * b, (std::vector<double>{4.0, 10.0, 18.0}));
    expect_values(b / a, (std::vector<double>{4.0, 2.5, 2.0}));
    expect_values(a + 2.0, (std::vector<double>{3.0, 4.0, 5.0}));
    expect_values(2.0 + a, (std::vector<double>{3.0, 4.0, 5.0}));
    expect_values(a - 2.0, (std::vector<double>{-1.0, 0.0, 1.0}));
    expect_values(2.0 - a, (std::vector<double>{1.0, 0.0, -1.0}));
    expect_values(a * 2.0, (std::vector<double>{2.0, 4.0, 6.0}));
    expect_values(12.0 / a, (std::vector<double>{12.0, 6.0, 4.0}));
}

TEST(Operators, UnaryAndMismatchedLengths)
{
    sparrow::primitive_array<int32_t> a{-1, 2, -3};
    sparrow::primitive_array<int32_t> b{1, 2};

    expect_values(+a, (std::vector<int32_t>{-1, 2, -3}));
    expect_values(-a, (std::vector<int32_t>{1, -2, 3}));
    EXPECT_THROW((void) sparrow::compute::eval(a + b), std::invalid_argument);
}

TEST(Expressions, BooleanInputsUseBitmapAwareAdaptor)
{
    sparrow::primitive_array<bool> values{true, false, true, false};

    expect_values(
        sparrow::compute::eval(!sparrow::compute::as_expression(values)),
        (std::vector<bool>{false, true, false, true})
    );
}

TEST(Math, CoreElementwiseFunctions)
{
    sparrow::primitive_array<double> values{-1.0, 0.0, 1.0, 4.0};
    sparrow::primitive_array<double> exponents{2.0, 2.0, 2.0, 0.5};
    sparrow::primitive_array<double> other{3.0, 4.0, 5.0, 6.0};
    sparrow::primitive_array<double> trig{-1.0, 0.0, 1.0};

    const auto expression = sparrow::compute::as_expression(values);
    expect_values(sparrow::compute::abs(expression), (std::vector<double>{1.0, 0.0, 1.0, 4.0}));
    expect_values(sparrow::compute::minimum(expression, sparrow::compute::as_expression(other)), (std::vector<double>{-1.0, 0.0, 1.0, 4.0}));
    expect_values(sparrow::compute::maximum(expression, sparrow::compute::as_expression(other)), (std::vector<double>{3.0, 4.0, 5.0, 6.0}));
    expect_values(sparrow::compute::clip(expression, 0.0, 1.0), (std::vector<double>{0.0, 0.0, 1.0, 1.0}));
    expect_values(sparrow::compute::sqrt(sparrow::compute::abs(expression)), (std::vector<double>{1.0, 0.0, 1.0, 2.0}));
    expect_values(sparrow::compute::pow(sparrow::compute::abs(expression), sparrow::compute::as_expression(exponents)), (std::vector<double>{1.0, 0.0, 1.0, 2.0}));
    expect_values(sparrow::compute::asin(sparrow::compute::sin(sparrow::compute::as_expression(trig))), (std::vector<double>{-1.0, 0.0, 1.0}));
    expect_values(sparrow::compute::floor(expression), (std::vector<double>{-1.0, 0.0, 1.0, 4.0}));
}

TEST(Math, FloatingPointFamilies)
{
    sparrow::primitive_array<double> positive{1.0, 2.0, 8.0};
    sparrow::primitive_array<double> angles{0.0, 0.5, 1.0};
    sparrow::primitive_array<double> rounded{-1.5, 0.4, 1.5};
    sparrow::primitive_array<double> legs{3.0, 4.0, 5.0};

    const auto exp_log = sparrow::compute::exp(sparrow::compute::log(sparrow::compute::as_expression(positive)));
    const auto hyperbolic = sparrow::compute::asinh(sparrow::compute::sinh(sparrow::compute::as_expression(angles)));
    const auto hypotenuse = sparrow::compute::hypot(sparrow::compute::as_expression(legs), sparrow::compute::as_expression(legs));

    for (std::size_t i = 0; i < positive.size(); ++i)
    {
        EXPECT_NEAR(exp_log(i), positive[i].value(), 1e-12);
        EXPECT_NEAR(hyperbolic(i), angles[i].value(), 1e-12);
        EXPECT_NEAR(hypotenuse(i), legs[i].value() * std::sqrt(2.0), 1e-12);
    }
    expect_values(sparrow::compute::round(sparrow::compute::as_expression(rounded)), (std::vector<double>{-2.0, 0.0, 2.0}));
}

TEST(Expressions, AllImportedOperatorsEvaluate)
{
    sparrow::primitive_array<int32_t> a{6, 5, 4};
    sparrow::primitive_array<int32_t> b{2, 2, 2};
    const auto lhs = sparrow::compute::as_expression(a);
    const auto rhs = sparrow::compute::as_expression(b);

    expect_values(sparrow::compute::eval(lhs + rhs), (std::vector<int32_t>{8, 7, 6}));
    expect_values(sparrow::compute::eval(lhs - rhs), (std::vector<int32_t>{4, 3, 2}));
    expect_values(sparrow::compute::eval(lhs * rhs), (std::vector<int32_t>{12, 10, 8}));
    expect_values(sparrow::compute::eval(lhs / rhs), (std::vector<int32_t>{3, 2, 2}));
    expect_values(sparrow::compute::eval(lhs % rhs), (std::vector<int32_t>{0, 1, 0}));
    expect_values(sparrow::compute::eval(+lhs),         (std::vector<int32_t>{6, 5, 4}));
    expect_values(sparrow::compute::eval(-lhs),         (std::vector<int32_t>{-6, -5, -4}));
    expect_values(sparrow::compute::eval(lhs & rhs),    (std::vector<int32_t>{2, 0, 0}));
    expect_values(sparrow::compute::eval(lhs | rhs),    (std::vector<int32_t>{6, 7, 6}));
    expect_values(sparrow::compute::eval(lhs ^ rhs),    (std::vector<int32_t>{4, 7, 6}));
    expect_values(sparrow::compute::eval(~lhs),             (std::vector<int32_t>{~6, ~5, ~4}));

    const auto equal = sparrow::compute::equal(lhs, rhs);
    const auto not_equal = sparrow::compute::not_equal(lhs, rhs);
    expect_values(sparrow::compute::eval(not_equal), (std::vector<bool>{true, true, true}));
    expect_values(sparrow::compute::eval(lhs < rhs),    (std::vector<bool>{false, false, false}));
    expect_values(sparrow::compute::eval(lhs <= rhs), (std::vector<bool>{false, false, false}));
    expect_values(sparrow::compute::eval(lhs > rhs), (std::vector<bool>{true, true, true}));
    expect_values(sparrow::compute::eval(lhs >= rhs), (std::vector<bool>{true, true, true}));
    expect_values(sparrow::compute::eval(equal), (std::vector<bool>{false, false, false}));
    expect_values(sparrow::compute::eval(sparrow::compute::not_equal(lhs, rhs)), (std::vector<bool>{true, true, true}));
    expect_values(sparrow::compute::eval((lhs > rhs) && not_equal), (std::vector<bool>{true, true, true}));
    expect_values(sparrow::compute::eval((lhs < rhs) || sparrow::compute::equal(lhs, rhs)), (std::vector<bool>{false, false, false}));
    expect_values(sparrow::compute::eval(!(lhs > rhs)), (std::vector<bool>{false, false, false}));
}

TEST(Math, AllImportedFunctionsEvaluate)
{
    sparrow::primitive_array<double> positive{0.5, 1.0, 2.0};
    sparrow::primitive_array<double> other{1.5, 2.0, 3.0};
    sparrow::primitive_array<double> unit{0.25, 0.5, 0.75};
    sparrow::primitive_array<double> infinite{
        0.0,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()
    };
    const auto a = sparrow::compute::as_expression(positive);
    const auto b = sparrow::compute::as_expression(other);
    const auto u = sparrow::compute::as_expression(unit);
    const auto check = [](const auto& expression) {
        EXPECT_EQ(sparrow::compute::eval(expression).size(), 3u);
    };

    check(sparrow::compute::abs(-a));
    check(sparrow::compute::fabs(-a));
    check(sparrow::compute::fmod(b, a));
    check(sparrow::compute::remainder(b, a));
    check(sparrow::compute::fma(a, b, u));
    check(sparrow::compute::fmax(a, b));
    check(sparrow::compute::fmin(a, b));
    check(sparrow::compute::fdim(a, b));
    check(sparrow::compute::maximum(a, b));
    check(sparrow::compute::minimum(a, b));
    check(sparrow::compute::clip(b, 1.75, 2.5));
    check(sparrow::compute::sign(-a));
    check(sparrow::compute::exp(a));
    check(sparrow::compute::exp2(a));
    check(sparrow::compute::expm1(a));
    check(sparrow::compute::log(a));
    check(sparrow::compute::log10(a));
    check(sparrow::compute::log2(a));
    check(sparrow::compute::log1p(a));
    check(sparrow::compute::pow(a, b));
    check(sparrow::compute::sqrt(a));
    check(sparrow::compute::cbrt(a));
    check(sparrow::compute::hypot(a, b));
    check(sparrow::compute::sin(u));
    check(sparrow::compute::cos(u));
    check(sparrow::compute::tan(u));
    check(sparrow::compute::asin(u));
    check(sparrow::compute::acos(u));
    check(sparrow::compute::atan(u));
    check(sparrow::compute::atan2(a, b));
    check(sparrow::compute::sinh(u));
    check(sparrow::compute::cosh(u));
    check(sparrow::compute::tanh(u));
    check(sparrow::compute::asinh(u));
    check(sparrow::compute::acosh(b));
    check(sparrow::compute::atanh(u));
    check(sparrow::compute::erf(u));
    check(sparrow::compute::erfc(u));
    check(sparrow::compute::tgamma(a));
    check(sparrow::compute::lgamma(a));
    check(sparrow::compute::ceil(u));
    check(sparrow::compute::floor(u));
    check(sparrow::compute::trunc(u));
    check(sparrow::compute::round(u));
    check(sparrow::compute::nearbyint(u));
    check(sparrow::compute::rint(u));
    check(sparrow::compute::isfinite(sparrow::compute::as_expression(infinite)));
    check(sparrow::compute::isinf(sparrow::compute::as_expression(infinite)));
    check(sparrow::compute::isnan(sparrow::compute::as_expression(infinite)));
    check(sparrow::compute::where(a < b, a, b));

    expect_values(sparrow::compute::eval(sparrow::compute::sign(-a)), (std::vector<double>{-1.0, -1.0, -1.0}));
    expect_values(sparrow::compute::eval(sparrow::compute::isfinite(sparrow::compute::as_expression(infinite))), (std::vector<bool>{true, false, false}));
    expect_values(sparrow::compute::eval(sparrow::compute::where(a < b, a, b)), (std::vector<double>{0.5, 1.0, 2.0}));
}
