#pragma once

#include <functional>

#include <sparrow/primitive_array.hpp>

#include "sparrow-compute/math.hpp"

// These overloads intentionally live in sparrow so argument-dependent lookup
// finds them for primitive_array expressions without a using-directive.
namespace sparrow
{
    namespace detail
    {
        template <primitive_type T>
        [[nodiscard]] auto as_compute_operand(const primitive_array<T>& value)
        {
            return compute::as_expression(value);
        }

        template <class T>
        [[nodiscard]] const T& as_compute_operand(const T& value)
        {
            return value;
        }

        template <class Operation, class L, class R>
        [[nodiscard]] auto apply_binary_operation(
            Operation operation,
            const L& lhs,
            const R& rhs
        )
        {
            return std::invoke(
                operation,
                as_compute_operand(lhs),
                as_compute_operand(rhs)
            );
        }
    }  // namespace detail

    // C++ requires a distinct overload declaration for every operator token.
    // Keep those ADL entry points thin and centralize the conversion to an
    // xtensor-compatible expression in the helpers above.
    template <primitive_type T>
    [[nodiscard]] auto operator+(const primitive_array<T>& a, const primitive_array<T>& b)
    {
        return detail::apply_binary_operation(std::plus{}, a, b);
    }

    template <primitive_type T>
    [[nodiscard]] auto operator-(const primitive_array<T>& a, const primitive_array<T>& b)
    {
        return detail::apply_binary_operation(std::minus{}, a, b);
    }

    template <primitive_type T>
    [[nodiscard]] auto operator*(const primitive_array<T>& a, const primitive_array<T>& b)
    {
        return detail::apply_binary_operation(std::multiplies{}, a, b);
    }

    template <primitive_type T>
    [[nodiscard]] auto operator/(const primitive_array<T>& a, const primitive_array<T>& b)
    {
        return detail::apply_binary_operation(std::divides{}, a, b);
    }

#define SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR(symbol, operation)                                \
    template <primitive_type T>                                                                   \
    [[nodiscard]] auto operator symbol(const primitive_array<T>& a, const T& b)                  \
    {                                                                                              \
        return detail::apply_binary_operation(operation{}, a, b);                                \
    }                                                                                              \
    template <primitive_type T>                                                                   \
    [[nodiscard]] auto operator symbol(const T& a, const primitive_array<T>& b)                  \
    {                                                                                              \
        return detail::apply_binary_operation(operation{}, a, b);                                \
    }

    SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR(+, std::plus)
    SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR(-, std::minus)
    SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR(*, std::multiplies)
    SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR(/, std::divides)

#undef SPARROW_COMPUTE_DEFINE_SCALAR_OPERATOR

    template <primitive_type T>
    [[nodiscard]] auto operator+(const primitive_array<T>& a)
    {
        return +compute::as_expression(a);
    }

    template <primitive_type T>
    [[nodiscard]] auto operator-(const primitive_array<T>& a)
    {
        return -compute::as_expression(a);
    }
}  // namespace sparrow
