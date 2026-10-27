# sparrow-compute

Header-only lazy xtensor expressions for `sparrow::primitive_array`.

```cpp
#include <sparrow-compute.hpp>

sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
sparrow::primitive_array<double> b{4.0, 5.0, 6.0};
auto result = sparrow::compute::eval(sparrow::compute::sqrt((a + b) / 2.0));
```

`sparrow-compute.hpp` exposes the whole public API as a single include. The
individual headers under `sparrow-compute/` remain available for finer-grained
includes.

`a + b` produces a lazy xtensor-compatible expression; use `eval` to materialize
a new Sparrow array. To start a named xtensor math expression from a single
array, wrap it with `as_expression(a)`, e.g. `sqrt(as_expression(a))`. Inputs
must be non-null; expression result types follow xtensor's functors.

The library version is available from
`sparrow::compute::SPARROW_COMPUTE_VERSION_MAJOR`,
`sparrow::compute::SPARROW_COMPUTE_VERSION_MINOR`, and
`sparrow::compute::SPARROW_COMPUTE_VERSION_PATCH`.

## Releasing

Releases are published from the GitHub Actions **Release** workflow. It bumps
`include/sparrow-compute/config/sparrow_compute_version.hpp`, tags the release
as `X.Y.Z` (or `X.Y.Z_rN` for prereleases), pushes the tag, and creates the
GitHub release.
