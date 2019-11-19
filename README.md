# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation**                                                               | **Build Status**                                                                                | **Digital Object Identifier**  |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3245497.svg)](https://doi.org/10.5281/zenodo.3245497) |

**TensorOperations v2.0.0 represents a significant update and rewrite from previous versions.**

* Tensoroperations.jl now exports an `ncon` method, familiar in the quantum tensor network community and mostly compatible with e.g. [arXiv:1402.0939](https://arxiv.org/abs/1402.0939). Unlike the `@tensor` which has been at the heart of TensorOperations.jl, the `ncon` analyzes the network at runtime, and as a consequence has a non-inferrable output. On the other hand, this allows to use dynamical index specifications which are not known at compile time. There is also an `@ncon` macro which uses the same format and also allows for dynamical index specifications, but has the advantage that it adds a hook into the global LRU cache where temporary objects are stored and recycled.

* TensorOperations.jl now supports `CuArray` objects via the NVidia's CUTENSOR library, which is wrapped in CuArrays.jl. This requires that the latter is also loaded with `using CuArrays`. `CuArray` objects can directly be used in the existing calls and macro environments like `@tensor` and `@ncon`. However, no operation should try to mix a normal `Array` and a `CuArray`. There is also a new `@cutensor` macro which will transform all array objects to the GPU and perform the contractions and permutations there. Objects are moved to the GPU when they are first needed, so that transfer times of later objects can coincide with computation time for operations on earlier objects.

* TensorOperations.jl now has a `@notensor` macro to indicate that a block within an `@tensor` environment (or `@tensoropt` or `@cutensor`) should be left alone and contains valid Julia code that should not be transformed.
 
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/TensorOperations.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/TensorOperations.jl/stable

[travis-img]: https://travis-ci.org/Jutho/TensorOperations.jl.svg?branch=master
[travis-url]: https://travis-ci.org/Jutho/TensorOperations.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/Jutho/TensorOperations.jl?svg=true&branch=master
[appveyor-url]: https://ci.appveyor.com/project/jutho/tensoroperations-jl/branch/master

[codecov-img]: https://codecov.io/gh/Jutho/TensorOperations.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorOperations.jl

[coveralls-img]: https://coveralls.io/repos/github/Jutho/TensorOperations.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/Jutho/TensorOperations.jl


### Code Example

TensorOperations.jl is mostly used through the `@tensor` macro which allows one to express a given operation in terms of [index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format, a.k.a. [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) (using Einstein's summation convention).
```julia
using TensorOperations
α=randn()
A=randn(5,5,5,5,5,5)
B=randn(5,5,5)
C=randn(5,5,5)
D=zeros(5,5,5)
@tensor begin
    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
end
```
In the second to last line, the result of the operation will be stored in the preallocated array `D`, whereas the last line uses a different assignment operator `:=` in order to define and allocate a new array `E` of the correct size. The contents of `D` and `E` will be equal.

For more information, please see the docs.
