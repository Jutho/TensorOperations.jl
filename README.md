<img src="https://github.com/Jutho/TensorOperations.jl/blob/master/docs/src/assets/logo.svg" width="150">
# TensorOperations.jl
Fast tensor operations using a convenient Einstein index notation.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/TensorOperations.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/TensorOperations.jl/latest
[ci-img]: https://github.com/Jutho/TensorOperations.jl/workflows/CI/badge.svg
[ci-url]:
  https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3ACI
[ci-julia-nightly-img]:
  https://github.com/Jutho/TensorOperations.jl/workflows/CI%20(Julia%20nightly)/badge.svg
[ci-julia-nightly-url]:
  https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22
[codecov-img]:
  https://codecov.io/gh/Jutho/TensorOperations.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorOperations.jl
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.3245497.svg
[doi-url]: https://doi.org/10.5281/zenodo.3245497
[downloads-img]:
  https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TensorOperations
[downloads-url]: https://pkgs.genieframework.com?packages=TensorOperations

|                             **Documentation**                             |                                                      **Build Status**                                                       |
| :-----------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] [![][codecov-img]][codecov-url] |

| **Digital Object Identifier** |                         **Downloads**                         |
| :---------------------------: | :-----------------------------------------------------------: |
|  [![DOI][doi-img]][doi-url]   | [![TensorOperations Downloads][downloads-img]][downloads-url] |

## What's new in v4

- Moved CUDA to a package extension, to avoid unnecessary dependencies for Julia versions >= 1.9

- The cache for temporaries has been removed, but support for something similar is now provided through explicit allocating and freeing calls within the macro.

- The interface for custom types has been changed and thoroughly documented, making it easier to know what to implement. This has as a consequence that more general element types of tensors are now also possible.

- There is a new interface to work with backends, to allow for dynamic switching between different implementations of the TensorOperations interface.

- The `@tensor` macro now accepts keyword arguments to facilitate a variety of options that help with debugging, contraction cost and type wrapping.

- Some support for Automatic Differentiation has been added by adding reverse-mode chainrules.

> **WARNING:** TensorOperations 4.0 contains breaking changes and is in general incompatible with previous versions.

### Code example

TensorOperations.jl is mostly used through the `@tensor` macro which allows one
to express a given operation in terms of
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format,
a.k.a. [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
(using Einstein's summation convention).

```julia
using TensorOperations
α = randn()
A = randn(5, 5, 5, 5, 5, 5)
B = randn(5, 5, 5)
C = randn(5, 5, 5)
D = zeros(5, 5, 5)
@tensor begin
    D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

In the second to last line, the result of the operation will be stored in the
preallocated array `D`, whereas the last line uses a different assignment
operator `:=` in order to define and allocate a new array `E` of the correct
size. The contents of `D` and `E` will be equal.

For more information, please see the documentation.
