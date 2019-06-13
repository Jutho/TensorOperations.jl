# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation**                                                               | **Build Status**                                                                                | **Digital Object Identifier**  |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3245497.svg)](https://doi.org/10.5281/zenodo.3245497) |

**TensorOperations v1.0.0 represents a significant rewrite from previous versions.**

While the exported API was left mostly unchanged, there are a few
breaking changes, especially in the function syntax.

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
