# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] |

**TensorOperations v1.0.0 represents a significant rewrite from previous versions.**

While the exported API was left mostly unchanged, there are a few
breaking changes, especially in the function syntax.

**TensorOperations v1.0.0 is still work in progress and is not yet released. The old Readme
for the current version of TensorOperations (v0.7) follows below:**

## What's new

- Addition of a `@tensoropt` macro that will optimize the contraction order of any product of tensors `A[...]*B[...]*C[...]*...` (see below for usage instructions and [this paper](https://doi.org/10.1103/PhysRevE.90.033315) for more details).
- The `@tensor` macro will reorganize the contraction order of the so-called NCON style of specifying indices is respected, i.e. all contracted indices are labelled by positive integers and all uncontracted indices are specified by negative integers. In that case, tensors will be contracted in such an order that indices with smaller integer label will
be contracted first
- Better overall type stability, both in the `@tensor(opt)` environment and with the function based approach. Even the simple function based syntax can now be made type stable by specifing the indices using tuples.
- Fully compatible with Julia v0.7/v1.0 (v0.6 no longer supported).

## Installation

Install with the package manager, `pkg> add TensorOperations`.

## Philosophy

The TensorOperations.jl package provides a convenient macro interface to specify tensor operations such as tensor contractions and index permutations via index notation. The index notation is analyzed at compile time and the resulting operations are then computed by efficient (cache-friendly) methods. Implementations are provided for arbitrary `StridedArray` instances, i.e. dense arrays whose data is layed out in memory in a strided fashion. In particular, TensorOperations.jl deals with objects of Julia's built-in `Array` and `SubArray` type. It can however also easily be extended to custom user defined types by overloading a miminal set of functions.

## Tensor operations

TensorOperations.jl is centered around 3 basic tensor operations:

1. **addition:** Add a (possibly scaled version of) one array to another array, where the indices of the both arrays might appear in different orders. This operation combines normal array addition and index permutation. It includes as a special case copying one array into another with permuted indices, and provides a cache-friendly (and thus more efficient) alternative to `permutedims` from Julia Base.

2. **trace or inner contraction:** Perform a trace/contraction over pairs of indices of an array, where the result is a lower-dimensional array.

3. **contraction :** Performs a general contraction of two tensors, where some indices of one array are paired with corresponding indices in a second array. Contains as special case the outer product where no indices are contracted and a new array is created with a number of dimensions that is the sum of the number of dimensions of the two original arrays.

### Index notation

The prefered way to specify (a sequence of) tensor operations is by using the `@tensor` macro which accepts an index notation format, which includes Einstein's summation convention. This can most easily be explained using a simple example:

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

In the second to last line, the result of the operation will be stored in the preallocated array `D`, whereas the last line uses a different assignment operator `:=` in order to define a new array `E` of the correct size. The contents of `D` and `E` will be equal.

Following Einstein's summation convention, the result is computed by first tracing/contracting the 3rd and 5th index of array `A`. The resulting array will then be contracted with array `B` by contracting its 2nd index with the last index of `B` and its last index with the first index of `B`. The resulting array has three remaining indices, which correspond to the indices `a` and `c` of array `A` and index `b` of array `B` (in that order). To this, the array `C` (scaled with `α`) is added, where its first two indices will be permuted to fit with the order `a,c,b`. The result will then be stored in array `D`, which requires a second permutation to bring the indices in the requested order `a,b,c`.

In this example, the labels were specified by arbitrary letters or even longer names. Any valid variable name is valid as a label. Note though that these labels are never interpreted as existing Julia variables, but rather are converted into symbols by the `@tensor` macro. This means, in particular, that the specific tensor operations defined by the code inside the `@tensor` environment are completely specified at compile time. Alternatively, one can also choose to specify the labels using literal integer or character constants, such that also the following code specifies the same operation as above. Finally, it is also allowed to use primes to denote different indices

```julia
@tensor D[å,ß,c'] = A[å,1,'f',c','f',2]*B[2,ß,1] + α*C[c',å,ß]
```

The index pattern is analyzed at compile time and wrapped in appropriate types such that the result of the operation can be computed with a minimal number of temporaries. The use of `@generated` functions further enables to move as much of the label analysis to compile time. You can read more about these topics in the section "Implementation" below.

By default, a contraction of several tensors `A[a,b,c,d,e]*B[b,e,f,g]*C[c,f,i,j]*...` will be evaluted using pairwise contractions from left to right, i.e. as `( (A[a,b,c,d,e] * B[b,e,f,g]) * C[c,f,i,j]) * ...`. However, if one respects the so-called [NCON](https://arxiv.org/abs/1402.0939) style of specifying indices, i.e. positive integers for the contracted indices and negative indices for the open indices, the different factors will be reordered and so that the pairwise tensor contractions contract over indices with smaller integer label first. For example, `D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]` will be evaluated as `(A[-1,3,1,-2,2]*C[1,4,-4,-3])*B[3,2,4,-5]`. Furthermore, in that case the indices of the output tensor (`D` in this case) do not need to be specified, and will be chosen as `(-1,-2,-3,-4,-5)`. Any other order is of course still possible by just specifying it.

Furthermore, there is a `@tensoropt` macro which will optimize the contraction order to minimize the total number of multiplications (cost model might change or become choosable in the future). The optimal contraction order will be determined at compile time and will be hard coded in the macro expansion. The cost/size of the different indices can be specified in various ways, and can be integers or some arbitrary polynomial of an abstract variable, e.g. `χ`. In the latter case, the optimization assumes the assymptotic limit of large `χ`.

```julia
@tensoropt D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]
# cost χ for all indices (a,b,c,d,e,f)
@tensoropt (a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]
# cost χ for indices a,b,c,e, other indices (d,f) have cost 1
@tensoropt !(a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]
# cost 1 for indices a,b,c,e, other indices (d,f) have cost χ
@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]
# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
```
The optimal contraction tree as well as the associated cost can be obtained by
```julia
@optimalcontractiontree C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]
```
where the cost of the indices can be specified in the same various ways as for `@tensoropt`.

### Functions

The elementary tensor operations can also be accessed via functions, mainly for compatibility with older versions of this toolbox. The function-based syntax is also required when the contraction pattern is not known at compile time but is rather determined dynamically.

These functions come in a mutating and non-mutating version. The mutating versions mimick the argument order of some of the BLAS functions, such as `blascopy!`, `axpy!` and `gemm!`. Symbols `A` and `B` always refer to input arrays, whereas `C` is used to denote the array where the result will be stored. The greek letters `α` and `β` denote scalar coefficients.

- `tensorcopy!(A, labelsA, C, labelsC)`

  Copies the data of array `A` to `C`, according to the label pattern specified by `labelsA` and `labelsC`. The result of this function is equivalent to `permutedims!(C,A,p)` where `p` is the permutation such that `labelsC=labelsA[p]`.

- `tensoradd!(α, A, labelsA, β, C, labelsC)`

  Replaces `C` with `β C + α A` with an additional permutation of the data in array `A` according to the order to go from `labelsA` to `labelsC`.

- `tensortrace!(α, A, labelsA, β, C, labelsC)`

  Replaces C with `β C + α A` where some of the indices of `A` are traced/contracted over, by assigning them unique labels in `labelsA`. Every label should appear exactly twice in the union of `labelsA` and `labelsC`, either twice in `labelsA` (for indices that need to be contracted) or once in both arguments, for indicating the order in which the result of tracing `A` needs to be added to `C`.

- `tensorcontract!(α, A, labelsA, conjA, B, labelsB, conjB, β, C, labelsC; method=:BLAS)`

  Replaces C with `β C + α A * B`, where some indices of array `A` are contracted with corresponding indices in array `B` by assigning them identical labels in the iterables `labelsA` and `labelsB`. The arguments `conjA` and `conjB` should be of type `Char` and indicate whether the data of arrays `A` and `B`, respectively, need to be conjugated (value `'C'`) or not (value `'N'`). Every label should appear exactly twice in the union of `labelsA`, `labelsB` and `labelsC`, either in the intersection of `labelsA` and `labelsB` (for indices that need to be contracted) or in the interaction of either `labelsA` or `labelsB` with `labelsC`, for indicating the order in which the open indices should be match to the indices of the output array `C`.

  There is an optional keyword argument `method` whose value can be `:BLAS` or `:native`. The first option creates temporary copies of `A`, `B` and the result where the indices are permuted such that the contractions become equivalent to a single matrix multiplication, which is typically handled by BLAS. This is often the fastest approach and therefore the default value, but it does require sufficient memory and there is some overhead in allocating new memory (e.g. when doing this many times in a loop). In case `method` is set to `:native`, a Julia function is called that performs the contraction without creating tempories, with special attention to cache-friendliness for maximal efficiency. See the "Implementation" section below for additional information.

- `tensorproduct!(α, A, labelsA, B, labelsB, β, C, labelsC)`

  Replaces C with `β C + α A * B` without any indices being contracted.

The non-mutating functions are simpler in not allowing scalar coefficients and conjugation. They also take a default value for the labels of the output array if these are not specified. They are simply called as:

- `C = tensorcopy(A, IA, IC=IA)`
- `C = tensoradd(A, IA, B, IB, IC=IA)`
- `C = tensortrace(A, IA, IC=unique2(IA))`

  where `unique2` is an auxiliary function that eliminates any label that appears twice in `IA`.

- `C = tensorcontract(A, IA, B, IB, IC=symdiff(IA,IB); method=:BLAS)`

- `C = tensorproduct(A, IA, B, IB, IC=union(IA,IB))`

For type stability, the functions for tensor operations always assume the result to be an array, even if the result is a single number, e.g. when tracing all indices of an array or contracting all indices between two arrays. The auxiliary function `scalar` can be used to extract the single non-zero component of a zero-dimensional array and store it in a non-container variable:

- `C = scalar(A)`

  Returns the single element of a length 1 array, e.g. a zero-dimensional array or any higher-dimensional array which has `size=(1,1,1,1,...)`.

## Implementation

### Building blocks

Under the hood, the implementation is again centered around the basic unit operations: addition, tracing and contraction. These operations are implemented for arbitrary instances of type `StridedArray` with arbitrary element types. The implementation can easily be extended to user defined types, especially if they just wrap multidimensional data with a strided memory storage, as explained below.

The building blocks resemble the functions discussed above, but have a different interface and are more general. They are used both by the functions as well as by the `@tensor` macro, as discussed below. We here just summarize their functionality and further discuss the implementation for strided data in the next section. Note that these functions are not exported.

- `add!(α, A, conjA, β, C, indCinA)`

  Implements `C = β*C+α*permute(op(A))` where `A` is permuted according to `indCinA` and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains as nth entry the dimension of `A` associated with the nth dimension of `C`.

- `trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)`

  Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced, according to `indCinA`, `cindA1` and `cindA2`, and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains as nth entry the dimension of `A` associated with the nth dimension of `C`. The partial trace is performed by contracting dimension `cindA1[i]` of `A` with dimension `cindA2[i]` of `A` for all `i in 1:length(cindA1)`.

- `contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, [method])`

  Implements `C = β*C+α*contract(op(A),op(B))` where `A` and `B` are contracted according to `oindA`, `cindA`, `oindB`, `cindB` and `indCinoAB`. The operation `op` acts as `conj` if `conjA` or `conjB` equal `Val{:C}` or as the identity map if `conjA` (`conjB`) equal `Val{:N}`. The dimension `cindA[i]` of `A` is contracted with dimension `cindB[i]` of `B`. The `n`th dimension of C is associated with an uncontracted (open) dimension of `A` or `B` according to `indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA]` with `NoA=length(oindA)` the number of open dimensions of `A`.

  The optional argument `method` specifies whether the contraction is performed using BLAS matrix multiplication by specifying `Val{:BLAS}`, or using a native algorithm by specifying `Val{:native}`. The native algorithm does not copy the data but is typically slower. The BLAS-based algorithm is chosen by default, if the element type of the output array is in `Base.LinAlg.BlasFloat`.

### Implementation for `StridedArray`

`TensorOperations.jl` provides implementation for any Julia `StridedArray{T,N}` for arbitrary element type `T` and arbitrary dimensionality `N`. The assumption that the multidimensional data has strided memory storage is crucial to the chosen implementation. It is generically not possible to simultaneously access the memory of the different arrays (`A` and `C` for `add!` and `trace!`, or `A`, `B` and `C` for `contract!`) in a cache-optimal way. Special care is given to cache-friendliness of the implementation by using a cache-oblivious divide-and-conquer strategy.

For `add!`, `trace!` and the native implementation of `contract!`, the problem is recursively divided into smaller blocks by slicing along those dimensions which correspond to the largest strides for all of the arrays. When the subproblem reaches a sufficiently small size, it is evaluated by a separate kernel using a set of nested for loops. The implementation depends heavily on metaprogramming and Julia's unique `@generated` functions to implement this strategy efficiently for any dimensionality. The minimal problem size is a constant which could be tuned depending on the cache size. The modularity of the implementation also allows to easily replace the kernels if better implementations would exist (e.g. when more SIMD features become available).

In order to deal with all types of `StridedArray` in a uniform way, and also to enhance the extensibility to user-defined arrays, `TensorOperations.jl` defines a new type `StridedData{N,T,C}`. This `immutable` wraps the strided data as a `data::Vector{T}`, which should be thought of as a memory pointer to the relevant memory region. It also includes a field `start::Int` such that `data[start]` is the first item of the data and a field `strides::NTuple{N,Int}` that defines how to access the other elements of the multidimensional data. Furthermore, it has a type parameter `C` that specifies whether the data (`C=:N`) or the conjugated data (`C=:C`) should be used. Note that `StridedData` does not include the dimensionality, this is always specified separately. Within the recursive divide-and-conquer algorithms, `StridedData` groups the set of arguments that remains constant, whereas the dimensionality and an additional offset are updated when dividing the problem into smaller subproblems.

We now provide some additional details on the specific implementation of the three building blocks:

- `add!`

  The `add!` operation corresponds schematically to `C = β C + α perm(op(A))` where the dimensions of `A` are permuted with respect to those of `C`. This operation generalizes the `axpy!` operation of BLAS to multidimensional arrays where the order of the different dimensions in both arrays can be different. The data in `A` and `C` are wrapped in a `StridedData` instance, and then passed on to `add_rec!` which implements the recursive divide-and-conquer strategy.

  Copying one array into another is a special case of addition corresponding to the choices `β=0` and `α=1`. Rather than providing a different implementation, special values `0` or `1` for `α` and/or `β` are intercepted early on and replaced by special singleton types `Zero()` and `One()`. An auxiliary function `axpby` which represents the operation `α*x + β*y` together with Julia's multiple dispatch is then exploited in order to make sure that no unnecessary calculations are performed when multiplying/adding those special values.

- `trace!`

  Tracing corresponds to the case where one or more pairs of dimensions of a higher-dimensional array `A` are traced/contracted and the result is added/copied to a lower-dimensional array `C`. While addition could be seen as a special case of this operation with zero pairs of contracted dimensions, we did not find a way of expressing this special case with zero overhead. Therefore, these two operations have a separate though completely analogous implementation.

- `contract!`

  For `contract!`, a native approach using the same divide-and-conquer strategy is also implemented. For big arrays whose element type is either `Float32`, `Float64`, `Complex64` or `Complex128` (the so-called `Base.LinAlg.BlasFloat` family), it is typically faster to rewrite the problem as a matrix multipliciation problem to be handled by the heavily tuned algorithm in the BLAS library. Thereto, the default implementation of `contract!` will in that case use `add!` on the two input arrays `A` and `B` to copy them to a permuted form such that the contraction is equivalent to matrix multiplication. A final `add!` is then used to copy or add this result back onto the output array `C`. While typically faster, this approach does require the allocation of temporary arrays to store the matrix equivalent of `A`, `B` and `C`. However, it only performs this copy when necessary and directly used the original arrays `A`, `B` or `C` if possible.

### The `@tensor` macro and allocation

Within an environment `@tensor begin ... end`, the indexing brackets `[...]` act as an assignment of a set of labels to a variable, which should be a multidimensional array. The names used inside the brackets are not interpreted as variables but transformed into symbols to be used as labels. Alternative label choices can be literal integers or characters. The `@tensor` macro will transform any set of labels into an object of the singleton type `Indices{I}` where the labels are stored as a tuple in the type parameter `I`. The indexing expression `A[a,b,c]` is transformed into an expression `indexify(A,Indices{(:a,:b,:c)})`, where the `indexify` function associates the indices with the object `A` using a new type discussed below. The assignment of a tensor expression to an already existing object in the left hand side (corresponding to `=`, `+=` or `-=`) is transformed into a call to `deindexify!`. If the left hand side needs to be created and allocated (corresponding to `:=`), a different call to the functions `deindexify` is made. Those functions are elaborated on below. Finally, if the left hand side is not an index expression, because e.g. the tensor expression on right hand side evaluates to a zero-dimensional array, the right hand side is automatically wrapped into a `scalar` call, to extract the single entry and store it in the non-array object on the left hand side.

To make use of the full generality of the building blocks, the macro based syntax depends on the following types. The `indexify` function creates instances of the type `IndexedObject{I,C,A,T}`, which wrap a multidimensional object `object::A` (no restrictions on `A`) and a scalar coefficient `α::T`. The set of labels of the object is stored as a tuple in the type parameter `I`, whereas the type parameter `C` encodes the effect of conjugation. This means that conjugating an array and/or multiplying it with a scalar is a delayed or lazy operation: these operations are not evaluated directly but rather stored inside the field and the type parameters. A linear combination of several `IndexedObject` instances is also not evaluated immediately but rather stored in an object of the type `SumOfIndexedObjects{Os<:Tuple{Vararg{AbstractIndexedObject}}}`. Similarly, the multiplication of two `IndexedObject` instances (which leads to contraction depending on their labels) gives rise to an object of `ProductOfIndexedObjects{IA,IB,CA,CB,OA,OB,TA,TB}`.

Evaluation of these operations is postponed until they appear in a `deindexify(!)` call corresponding to an assignment to the left hand side, or when they appear in a further set of operations, leading to the creation (=allocation) of a temporary array. The following call thus works without allocating any temporary array:

```julia
@tensor D[a,b,c] += α*A[a,c,b] + B[a,d,b,d,c] - conj(C[c,b,a])
```

Upon evaluation, first `α*A` will be added to `D` using a call to `add!`, then a call to `trace!` will add `B` to this result, and a final call to `add!` will add `conj(C)` to this result. Because the labels are stored in the type parameters, a `@generated` function is used to transform the label patterns into the correct arguments for `add!` and `trace!` at _compile time_.

If one of the terms contains a simple contraction of two arrays as in

```julia
@tensor D[a,b,c] = α*A[a,c,d,e]*conj(B[d,b,e]) + β*conj(C[c,b,a])
```

then still no memory is allocated for storing the intermedate result of the contraction. However, temporaries will likely be allocated in the default BLAS-based `contract!` routine, as discussed in the previous section. As before, a `@generated` function takes care of generating the correct arguments at compile time.

Any more advanced operation does need to create tempory arrays to store intermediate results. In particular:

- A contraction of three or more objects is evaluated as a pairwise contraction from left to right. Use parentheses to force a specific contraction order. See the section 'Planned features' for future ideas to automate this process.
- If one or both arguments of a pairwise contraction is itself a linear combination of multidimensional arrays, then it will be evaluated first.

### Code structure and extensibility

The `src` folder of `TensorOperations.jl` contains four subfolders in which the code is organised. The folder `implementation` contains the various parts of the implementation. The file `stridedarray.jl` provides the functions `add!`, `trace!` and `contract!` for arrays of type `StridedArray`. This file contains the main code that needs to be implemented in order to support other, user-defined multidimensional objects. If those user types just wrap a set of strided data, the implementation should be a straightforward analog. The file `recursive.jl` implements the divide-and-conquer algorithm for objects of type `StridedData`, whereas `kernels.jl` contains the kernels acting on the smallest subproblem. `indices.jl` contains the functions for transforming label patterns occuring in the index notation into valid and useful arguments for the routines `add!`, `trace!` and `contract!`. `strides.jl` contain some `@generated` functions to format the stride information in a shape that facilitates he implementation of the blocking strategy.

The folders `functions` and `indexnotation` contain the necessary code to support the function-based syntax and the `@tensor` macro syntax respectively. In particular, the folder `indexnotation` contains one file for the macro, and one file for each of the special types `IndexedObject`, `SumOfIndexedObjects` and `ProductOfIndexedObjects` discussed above. Both the function and macro based syntax are completely general and should work for any multidimensional object, provided a corresponding implementation for `add!`, `trace!` and `contract!` is provided.

Finally, the folder `auxiliary` contains some auxiliary code, such as some metaprogramming tools, the `unique2` function which removes all elements of a list which appear more than once, the definition of the `StridedData` type, the elementary `axpby` operation and the definition of a dedicated `IndexError` type for reporting errors in the index notation. Finally, there is a file `stridedarray.jl` which provides some auxiliary abstraction to interface with the `StridedArray` type of Julia Base. It contains the following function definitions:

- `numind(A)`

  Returns the number of indices of a tensor-like object `A`, i.e. for a multidimensional array (`<:AbstractArray`) we have `numind(A) = ndims(A)`. Also works in type domain.

- `similar_from_indices(T, indices::Tuple{Vararg{Int}}, A, conjA=Val{:N})`

  Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)`, where the selection is specified by `indices` (which contains integer between `1` and `numind(A)`) and `op` is `conj` if `conjA=Val{:C}` or does nothing if `conjA=Val{:N}` (default).

- `similar_from_indices(T, indices::Tuple{Vararg{Int}}, A, B, conjA=Val{:N}, conjB={:N})`

  Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)` and `op(B)` concatenated, where the selection is specified by `indices` (which contains integers between `1` and `numind(A)+numind(B)` and `op` is `conj` if `conjA` or `conjB` equal `Val{:C}` or does nothing if `conjA` or `conjB` equal `Val{:N}` (default).

- `scalar(C)`

  Returns the single element of a tensor-like object with zero dimensions, i.e. if `numind(C)==0`.

In summary, to add support for a user-defined tensor-like type, the functions in the files `auxiliary/stridedarray.jl` and `implementation/stridedarray.jl` should be reimplemented. This should be rather straightforwarded if the type just wraps multidimensional data with a strided storage in memory.

## Planned features

The following features seem like interesting additions to the `TensorOperations.jl` package, and might therefore appear in the future.

- Implementation of a tensor contraction implementation that can immediately call the BLAS microkernels to act on small blocks of the arrays which are permuted into fixed size buffers. Or rather, possibility to use some of the recent efforts in this direction as backend (e.g. [TBLIS](https://github.com/devinamatthews/tblis), [HPTT](https://github.com/springer13/hptt) and [TCL](https://github.com/springer13/tcl)).
- Implementation of a `@rawindex` macro that translates an expression with index notation directly into a raw set of nested loops. While loosing the advantage of cache-friendlyness for large arrays, this can be advantageous for smaller arrays. In addition, this would allow for more general expressions where functions of array entries are computed without having to allocate a temporary array, or where three or more dimensions are simultaneously summed over.


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
