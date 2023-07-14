# Functions

The elementary tensor operations can also be accessed via functions, mainly for
compatibility with older versions of this toolbox. The function-based syntax is also
required when the contraction pattern is not known at compile time but is rather determined
dynamically.

The basic exposed interface, as listed below, makes use of any iterable `IA`, `IB` or `IC`
to denote labels of indices, in a similar fashion as when used in the context of
[`@tensor`](@ref). When making use of this functionality, in-place operations are no longer
supported, as these are reserved for the *expert mode*. Note that the return type is only
inferred when the labels are entered as tuples, and also `IC` is specified.

The expert mode exposes both mutating and non-mutating versions of these functions. In this
case, selected indices are determined through permutations, specified by `pA`, `pB` and
`pC`. In order to distinguish from the non-mutating version in *simple mode*, overlapping
functionality is distinguished by specializing on these permutations, which are required to
take a particular form of the type [`Index2Tuple`](@ref).

```@docs
Index2Tuple
```

The motivation for this particular convention for specifying permutations comes from the
fact that for many operations, it is useful to think of a tensor as a linear map or matrix,
in which its different indices are partioned into two groups, the first of which correspond
to the range of the linear map (the row index of the associated matrix), whereas the second
group corresponds to the domain of the linear map (the column index of the associated
matrix). This is most obvious for tensor contraction, which then becomes equivalent to
matrix multiplication (which is also how it is implemented by the `StridedBLAS` backend).
While less relevant for tensor permutations, we use this convention throughout for
uniformity and generality (e.g. for compatibility with libraries that always represent
tensors as linear maps, such as [TensorKit.jl](http://github.com/Jutho/TensorKit.jl)).

Note, finally, that only the expert mode call style exposes the ability to select custom
backends.

## Non-mutating functions

```@docs
tensorcopy
tensoradd
tensortrace
tensorcontract
tensorproduct
```

## Mutating functions

```@docs
tensorcopy!
tensoradd!
tensortrace!
tensorcontract!
tensorproduct!
```
