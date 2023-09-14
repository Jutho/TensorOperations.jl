<img src="https://github.com/Jutho/TensorOperations.jl/blob/master/docs/src/assets/logo.svg" width="150">

# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation** | **Digital Object Identifier** |
|:-----------------:|:-----------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] |

| **Build Status** | **Coverage** | **Quality assurance** | **Downloads** |
|:----------------:|:------------:|:---------------------:|:--------------|
| [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] |


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/TensorOperations.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/TensorOperations.jl/latest

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.3245497.svg
[doi-url]: https://doi.org/10.5281/zenodo.3245497

[ci-img]: https://github.com/Jutho/TensorOperations.jl/workflows/CI/badge.svg
[ci-url]:
  https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3ACI

[codecov-img]:
  https://codecov.io/gh/Jutho/TensorOperations.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorOperations.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[downloads-img]:
  https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TensorOperations
[downloads-url]: https://pkgs.genieframework.com?packages=TensorOperations

## What's new in v4

- The `@tensor` macro now accepts keyword arguments to facilitate a variety of options that help with debugging, contraction cost and backend selection.

- Experimental support for automatic differentiation has been added by adding reverse-mode chainrules.

- The interface for custom types has been changed and thoroughly documented, making it easier to know what to implement. This has as a consequence that more general element types of tensors are now also possible.

- There is a new interface to work with backends, to allow for dynamic switching between different implementations of the primitive tensor operations or between different strategies for allocating new tensor objects.

- The support for `CuArray` objects is moved to a package extension, to avoid unnecessary CUDA dependencies for Julia versions >= 1.9

- The cache for temporaries has been removed due to its inconsistent and intricate interplay with multithreading.
  However, the new feature of specifying custom allocation strategies can be used to experiment with novel cache-like behaviour in the future.

> **WARNING:** TensorOperations 4.0 contains seveal breaking changes and cannot generally be expected to be compatible with previous versions.

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

For more detailed information, please see the documentation.
