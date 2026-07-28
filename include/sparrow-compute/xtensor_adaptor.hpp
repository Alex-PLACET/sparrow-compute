#pragma once

#include <array>
#include <cstddef>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xnoalias.hpp>

#include <sparrow/arrow_interface/arrow_array_schema_proxy.hpp>
#include <sparrow/buffer/buffer_view.hpp>
#include <sparrow/layout/array_access.hpp>
#include <sparrow/primitive_array.hpp>

namespace sparrow::compute
{
    namespace detail
    {
        /// Get a raw pointer to the contiguous value buffer of a
        /// sparrow primitive array.
        template <sparrow::primitive_type T>
        [[nodiscard]] std::span<const T> view_data(const sparrow::primitive_array<T>& arr)
        {
            const auto& proxy = sparrow::detail::array_access::get_arrow_proxy(arr);
            return {
                proxy.buffers()[1].template data<T>() + static_cast<std::size_t>(proxy.offset()),
                static_cast<std::size_t>(proxy.length())
            };
        }
    }  // namespace detail

    /// Wrap a const sparrow primitive array as a 1-D xtensor adaptor.
    /// Const overload: the adaptor exposes the buffer read-only.
    template <sparrow::primitive_type T>
    [[nodiscard]] auto as_xtensor_view(const sparrow::primitive_array<T>& arr)
    {
        const auto data = detail::view_data(arr);
        return xt::adapt(
            data.data(),
            data.size(),
            xt::no_ownership(),
            std::array<std::size_t, 1>{data.size()}
        );
    }

    /// Build a sparrow primitive array from an xtensor expression
    /// with zero-copy evaluation.
    template <sparrow::primitive_type T, class E>
    [[nodiscard]] sparrow::primitive_array<T> to_sparrow(E&& expr)
    {
        const auto n = static_cast<std::size_t>(expr.shape()[0]);
        // xtensor overwrites every element below before ownership transfers to
        // primitive_array, so avoid value-initializing this output buffer.
        sparrow::u8_buffer<T> buffer(n, sparrow::uninitialized_t{});
        auto view = xt::adapt(
            buffer.data(),
            n,
            xt::no_ownership(),
            std::array<std::size_t, 1>{n}
        );
        xt::noalias(view) = std::forward<E>(expr);
        return sparrow::primitive_array<T>(std::move(buffer), n, /*nullable=*/false);
    }
}  // namespace sparrow::compute
