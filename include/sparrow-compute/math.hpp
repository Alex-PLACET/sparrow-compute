#pragma once

#include <xtensor/core/xmath.hpp>

#include "sparrow-compute/expression.hpp"

namespace sparrow::compute
{
    using xt::operator+;
    using xt::operator-;
    using xt::operator*;
    using xt::operator/;
    using xt::operator%;
    using xt::operator||;
    using xt::operator&&;
    using xt::operator!;
    using xt::operator&;
    using xt::operator|;
    using xt::operator^;
    using xt::operator~;
    using xt::operator!=;
    using xt::operator<;
    using xt::operator<=;
    using xt::operator>;
    using xt::operator>=;
    using xt::equal;
    using xt::not_equal;
    using xt::abs;
    using xt::fabs;
    using xt::fmod;
    using xt::remainder;
    using xt::fma;
    using xt::fmax;
    using xt::fmin;
    using xt::fdim;
    using xt::maximum;
    using xt::minimum;
    using xt::clip;
    using xt::sign;
    using xt::exp;
    using xt::exp2;
    using xt::expm1;
    using xt::log;
    using xt::log10;
    using xt::log2;
    using xt::log1p;
    using xt::pow;
    using xt::sqrt;
    using xt::cbrt;
    using xt::hypot;
    using xt::sin;
    using xt::cos;
    using xt::tan;
    using xt::asin;
    using xt::acos;
    using xt::atan;
    using xt::atan2;
    using xt::sinh;
    using xt::cosh;
    using xt::tanh;
    using xt::asinh;
    using xt::acosh;
    using xt::atanh;
    using xt::erf;
    using xt::erfc;
    using xt::tgamma;
    using xt::lgamma;
    using xt::ceil;
    using xt::floor;
    using xt::trunc;
    using xt::round;
    using xt::nearbyint;
    using xt::rint;
    using xt::isfinite;
    using xt::isinf;
    using xt::isnan;
    using xt::where;

}

namespace xt
{
    // xtensor reserves its existing operator!= overload for optional
    // expressions. This constrained overload supplies elementwise inequality
    // for its ordinary expression types.
    template <class E1, class E2>
        requires std::is_base_of_v<xexpression<E1>, E1>
              && std::is_base_of_v<xexpression<E2>, E2>
    [[nodiscard]] auto operator!=(const xexpression<E1>& lhs, const xexpression<E2>& rhs)
    {
        return not_equal(lhs.derived_cast(), rhs.derived_cast());
    }
}
