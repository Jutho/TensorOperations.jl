# TensorOperations.jl

[![Build Status](https://travis-ci.org/Jutho/TensorOperations.jl.svg)](https://travis-ci.org/Jutho/TensorOperations.jl) [![Coverage Status](https://img.shields.io/coveralls/Jutho/TensorOperations.jl.svg)](https://coveralls.io/r/Jutho/TensorOperations.jl)

Fast tensor operations using a convenient index notation or via memory friendly in-place methods.

## What's new
* Switched from cache aware to cache oblivious algorithms (divide and conquer).
* To use index notation, one now has to explicitly state `using TensorOperations, IndexNotation`

##Installation

Install with the package manager, `Pkg.add("TensorOperations")`

## Philosophy

The TensorOperations.jl package works with arbitrary StridedArray elements, i.e. dense arrays whose data is layed out in memory in a strided fashion. In particular, TensorOperations.jl deals with Julia's built-in Array and SubArray type. It can can easily be extened to custom user defined types, as explained below.

### Tensor operations

TensorOperations.jl is centered around 4 basic operations:

1. **tensorcopy:** Copy the data of one array into another, thereby possibly changing the order of the indices. This functionality replaces the built-in permutedims method with a cache-friendly (and thus more efficient) alternative
2. **tensoradd:** Add a (scaled version of) one array to another array, where the indices of the both arrays might appear in different orders. This operation combines normal array addition and index permutation.
3. **tensortrace:** Perform a trace/contraction over pairs of indices of an  array, where the result is a lower-dimensional array or a scalar (for an even-dimensional array where all indices are paired with another and traced over).
4. **tensorcontract:** Performs a general contraction of two tensors, where some indices of one array are paired with corresponding indices in a second array. Internal contractions should be handled separately with `tensortrace`.

These details of these operations are specified by assigning labels to the different indices of an array, where equal indices are identified and, in case of `tensortrace` or `tensorcontract` will be contracted.

### Index notation
These operations can be applied by calling the corresponding methods, as explained below, or using a convenient index notation format. This can most easily be explained using a simple example:

```
using TensorOperations
alpha=randn()
A=randn(5,5,5,5,5,5)
B=randn(5,5,5)
C=randn(5,5,5)
D=zeros(5,5,5)
D[l"a,b,c"]=A[l"a,e,f,c,f,g"]*B[l"g,b,e"]+alpha*C[l"c,a,b"]
```

The last line of code will first trace/contract the 3rd and 5th index of array A. The resulting array will then be contracted with array B by contracting its 2nd index with the last index of B and its last index with the first index of B. The resulting array has three remaining indices, which correspond to the indices `a` and `c` of array A and index `b` of array B (in that order). To this, the array C (scaled with alpha) is added, where its first two indices will be permuted to fit with the order `a,c,b`. The result will then be stored in array D, which requires a second permutation to bring the indices in the requested orer `a,b,c`. 

The use of an `l`-prefixed string creates an instance of the type `LabelList`, which is basically a wrapper for a `Vector{Symbol}`. The string is expected to be a comma seperated list of names, which will be converted to symbols. Indexing an array with such a LabelList will assign every name/symbol as a label for the corresponding index of the array. This behavior is obtained by defining `getindex` of an array indexed by a `LabelList` to return an instance of the `LabeledArray` type, which wraps the array data together with the labels. First, however, it is checked whether there are any reoccuring labels indicating that certain indices of the array need to be traced over. The operations `*` and `+` are then overloaded for `LabeledArray` instances so as to make appropriate calls to `tensorcontract` and `tensoradd`. Finally, `setindex!` on an array indexed by `LabelList` and with a `LabeledArray` on the right hand side results in a call to `tensorcopy!`, which copies the data of the right hand side into the array on the left hand side, possibly performing the corresponding permutation.

While convenient, the index notation creates a new temporary for every operation and might therefore not be the most efficient approach for large arrays. The mutating methods discussed below might be more appropriate in that case.

##Methods


There are two types of methods. The non-mutating methods are not typed and thus accept arbitrary input. They expect that arrays support the methods `similar` and `size` to create appropriate output arrays to store the result of the computation. They then call the corresponding mutating method supplying the newly created output array. Only the non-mutating methods are exported.

The details of the tensor operations are specified by providing labels for the indices of the arrays participating in the operation. Any iterable containing elements that can be compared for equality are suitable as list of labels. In particular, elements of type `Symbol`, `Char` or `Int` are suitable as labels, but so are most other types.

The mutating methods restrict the type of arrays to subtypes of `StridedArray`. Users can create new definitions of the mutating methods for their own custom array types, which will also make the non-mutating methods work (given that definitions for `similar` and `size` are also provided).

Note, finally, that the `LabeledArray` type is not a `StridedArray` and not even an `AbstractArray` (this can change in the future). `LabeledArray` is a datatype that wraps the arrays and the labels that can then be given as arguments to the methods below, and its only purpose is to make the index notation work.

### Scalar
For type stability, the methods for tensor operations always assume the result to be a type of array, even if the result is a single number, e.g. when tracing all indices of an array or contracting all indices between two arrays. The following method can surround a tensor operation to store the result in non-container variable:

* `scalar(A)`
  
  Returns the single element of a zero-dimensional array.


### Simple (non-mutating) methods

* `tensorcopy(A,labelsA[,outputlabels])`

  Creates a copy of array `A`, where the indices of `A` are labeled by the elements of the iterable `labelsA` and the labels of the copy are denoted by `outputlabels`. Both iterables should contain the same elements in a different order. The argument `outputlabels` is optional and its default value is `labelsA`, in which case a normal copy of `A` is returned.
  
  The result of this method is equivalent to `permutedims(A,p)` where `p` is the permutation such that `outputlabels=labelsA[p]`. The implementation of `tensorcopy` is however more efficient on average.
 
* `tensoradd(A,labelsA,B,labelsB[,outputlabels])`
  
  Returns the result of adding arrays `A` and `B` where the iterabels `labelsA` and `labelsB` denote how the array data should be permuted in order to be added. More specifically, the result of this method is equivalent to

  ```
  tensorcopy(A,labelsA,outputlabels)+tensorcopy(B,labelsB,outputlabels)
  ```
  
  but without creating the temporary permuted arrays. The default value for the optional argument `outputlabels` is `labelsA`.
  
* `tensortrace(A,labelsA[,outputlabels])`
  
  Trace or contract pairs of indices of array `A`, by assigning them an identical label in the iterable `labelsA`. The untraced indices, which are assigned a unique label, can be reordered according to the optional argument `outputlabels`. The default value corresponds to the order in which they appear. Note that only pairs of indices can be contracted, so that every label in `labelsA` can appear only once (for an untraced index) or twice (for an index in a contracted pair).
  
* `tensorcontract(A,labelsA,B,labelsB[,outputlabels;method])`
  
  Contract indices of array `A` with corresponding indices in array `B` by assigning them identical labels in the iterables `labelsA` and `labelsB`. The indices of the resulting array correspond to the indices that only appear in either `labelsA` or `labelsB` and can be ordered by specifying the optional argument `outputlabels`. The default is to have all open indices of array `A` followed by all open indices of array `B`. Note that inner contractions of an array should be handled first with `tensortrace`, so that every label can appear only once in `labelsA` or `labelsB` seperately, and once (for open index) or twice (for contracted index) in the union of `labelsA` and `labelsB`.
  
  There is an optional keyword argument `method` whose value can be `:BLAS` or `:native`. The first option creates temporary copies of `A`, `B` and the result where the indices are permuted such that the contractions become equivalent to a single matrix multiplication, which is typically handled by BLAS. This is often the fastest approach and therefore the default value, but it does require sufficient memory and there is some overhead in allocating new memory (e.g. when doing this many times in a loop). In case `method` is set to `:native`, a Julia function is called that performs the contraction without creating tempories, with special attention to cache-friendliness for maximal efficiency.
 
### Mutating methods

For the mutating methods, the argument order resembles some of the BLAS functions, such as `blascopy!`, `axpy!` and `gemm!`. Symbols `A` and `B` always refer to input arrays, whereas `C` is used to denote the array where the result will be stored. The mutating methods are defined for arrays of type `StridedArray` and also allow the output array `C` to have nontrivial strides.

* `tensorcopy!(A,labelsA,C,labelsC)`
  
  Copies the data of array `A` to `C`, according to the label pattern specified by `labelsA` and `labelsB`. The result of this method is equivalent to `permutedims!(C,A,p)` where `p` is the permutation such that `labelsC=labelsA[p]`. The implementation of `tensorcopy` is however more efficient on average, allows `C` to be an arbitrary `StridedArray` and also to have a different element type then `A` (e.g. to copy a `Float64` array to a `Complex128` array)
    
* `tensoradd!(alpha,A,labelsA,beta,C,labelsC)`
  
  Replaces `C` with `beta C + alpha A)` with an additional permutation of the data in array `A` according to the order to go from `labelsA` to `labelsC`.
  
* `tensortrace!(alpha,A,labelsA,beta,C,labelsC)`
  
  Replaces C with `beta C + alpha A` where some of the indices of `A` are traced/contracted over, by assigning them unique labels in `labelsA`. Every label should appear exactly twice in the union of `labelsA` and `labelsC`, either twice in `labelsA` (for indices that need to be contracted) or once in both arguments, for indicating the order in which the result of tracing `A` needs to be added to `C`.
  
* `tensorcontract!(alpha,A,labelsA,conjA,B,labelsB,conjB,beta,C,labelsC;method)`
  
  Replaces C with `beta C+alpha A * B`, where some indices of array `A` are contracted with corresponding indices in array `B` by assigning them identical labels in the iterables `labelsA` and `labelsB`. The arguments `conjA` and `conjB` should be of type `Char` and indicate whether the data of arrays `A` and `B`, respectively, need to be conjugated (value `C`) or not (value `N`). Every label should appear exactly twice in the union of `labelsA`, `labelsB` and `labelsC`, either in the intersection of `labelsA` and `labelsB` (for indices that need to be contracted) or in the interaction of either `labelsA` or `labelsB` with `labelsC`, for indicating the order in which the open indices should be match to the indices of the output array `C`.
   
  There is an optional keyword argument `method` whose value can be `:BLAS` or `:native`. The first option creates temporary copies of `A`, `B` and the result where the indices are permuted such that the contractions become equivalent to a single matrix multiplication, which is typically handled by BLAS. This is often the fastest approach and therefore the default value, but it does require sufficient memory and there is some overhead in allocating new memory (e.g. when doing this many times in a loop). In case `method` is set to `:native`, a Julia function is called that performs the contraction without creating tempories, with special attention to cache-friendliness for maximal efficiency.
  
### Internal methods

The mutating methods are implemented using a lot of metaprogramming corresponding to Julia's powerfull macro system and the extremely useful package `Cartesian.jl` by Tim Holy. Special care is given to cache-friendliness of the implementations by using cache-oblivious divide-and-conquer strategies. The parameters that determine the size of the base case are defined as constants in `TensorOperations.jl` and further tuning can improve the performance. The kernels for the base cases are seperately defined in kernels.jl (as macros) and can thus easily be replaced if better implementations would exist (e.g. when more SIMD features become available).

##Planned features


The following features seem like interesting additions to the TensorOperations.jl package, and might therefore appear in the future (not necessarily in this order)

* Further optimize cache-friendliness by changing the loop order in the contraction kernel depending such that the inner loops run over the array dimensions with the smallest strides. Further efficiency increase can also be obtained by differentiating between a number of different tensor operations that are all handled by `tensorcontract(!)`. To use BLAS terminology, depending on the `labelsA` and `labelsB` the result of `tensorcontract` can be a level 1 operation [a dot product (no open indices) or a tensor product (no contraction indices)], a level 2 operation [no open indices in one of the two input arrays] or a level 3 operation [a genuine contraction with open and contraction indices in both input arrays]. 

* Implementation of a a `:buffered` method for `tensorcontract(!)`, that can use a memory buffer in order to apply standard matrix multiplication on smaller subblocks of the input arrays without having to allocate new memory.

* Functionality to contract a large set of tensors, also called a tensor network, including a method to optimize over the contraction order along the lines of [arXiv:1304.6112v3](http://arxiv.org/abs/1304.6112v3).

* Implementation of a `@lazy` macro, that delays the evaluation of an index expression and computes the whole expression using an optimal strategy with a minimal number of temporaries, e.g. using a single buffer to store the required temporary arrays.