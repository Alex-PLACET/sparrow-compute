#pragma once

namespace sparrow::compute
{
    /// Semantic version of the sparrow-compute library.
    constexpr int SPARROW_COMPUTE_VERSION_MAJOR = 0;
    constexpr int SPARROW_COMPUTE_VERSION_MINOR = 1;
    constexpr int SPARROW_COMPUTE_VERSION_PATCH = 0;
}

namespace sparrow
{
    using compute::SPARROW_COMPUTE_VERSION_MAJOR;
    using compute::SPARROW_COMPUTE_VERSION_MINOR;
    using compute::SPARROW_COMPUTE_VERSION_PATCH;
}
