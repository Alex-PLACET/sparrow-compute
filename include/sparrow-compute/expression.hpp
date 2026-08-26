#pragma once

#include <type_traits>
#include <stdexcept>

#include <sparrow/buffer/dynamic_bitset/dynamic_bitset_view.hpp>
#include <sparrow/u8_buffer.hpp>

#include <xtensor/core/xexpression.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/utils/xexception.hpp>

#include "sparrow-compute/xtensor_adaptor.hpp"

namespace sparrow::compute
{
    /**
     * Expose a primitive array as a non-owning 1-D xtensor expression.
     * The input array must outlive every expression built from this adaptor.
     */
    template <sparrow::primitive_type T>
        requires(!std::is_same_v<T, bool>)
    [[nodiscard]] auto as_expression(const sparrow::primitive_array<T>& array)
    {
        return as_xtensor_view(array);
    }

    // Arrow booleans are bitmap-packed, whereas xtensor requires contiguous
    // bool values. Materialize this adaptor explicitly rather than treating the
    // bitmap as a bool pointer.
    [[nodiscard]] inline auto as_expression(const sparrow::primitive_array<bool>& array)
    {
        auto values = xt::xarray<bool>::from_shape({array.size()});
        for (std::size_t i = 0; i < array.size(); ++i)
        {
            values[i] = array[i].value();
        }
        return values;
    }

    /** Materialize an xtensor expression directly into a Sparrow value buffer. */
    template <class E>
    [[nodiscard]] auto eval(const xt::xexpression<E>& expression)
    {
        const auto& derived = expression.derived_cast();
        using value_type = typename std::decay_t<E>::value_type;
        try
        {
            if constexpr (std::is_same_v<value_type, bool>)
            {
                // Arrow bool values are bitmap-packed. Fill the final
                // u8_buffer through Sparrow's non-owning dynamic bitset view,
                // then transfer that buffer directly to primitive_array.
                const auto size = static_cast<std::size_t>(derived.size());
                const auto block_count = (size + 7u) / 8u;
                sparrow::u8_buffer<bool> data_buffer(block_count, false);
                sparrow::dynamic_bitset_view<std::uint8_t> bits(
                    reinterpret_cast<std::uint8_t*>(data_buffer.data()),
                    size
                );
                for (std::size_t i = 0; i < size; ++i)
                {
                    bits.set(i, derived[i]);
                }
                return sparrow::primitive_array<bool>(std::move(data_buffer), size, /*nullable=*/false);
            }
            else
            {
                return to_sparrow<value_type>(derived);
            }
        }
        catch (const xt::broadcast_error&)
        {
            throw std::invalid_argument("sparrow-compute expressions require equal-length inputs");
        }
    }
}
