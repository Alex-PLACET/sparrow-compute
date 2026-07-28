# sparrow-compute

Header-only lazy xtensor expressions for `sparrow::primitive_array`.

```cpp
#include <sparrow-compute/math.hpp>
#include <sparrow-compute/operators.hpp>

sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
sparrow::primitive_array<double> b{4.0, 5.0, 6.0};
auto result = sparrow::compute::eval(sparrow::compute::sqrt((a + b) / 2.0));
```

`a + b` produces a lazy xtensor-compatible expression; use `eval` to materialize
a new Sparrow array. To start a named xtensor math expression from a single
array, wrap it with `as_expression(a)`, e.g. `sqrt(as_expression(a))`. Inputs
must be non-null; expression result types follow xtensor's functors.
