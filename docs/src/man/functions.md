# Functions

The elementary tensor operations can also be accessed via functions, mainly for compatibility with older versions of this toolbox.
The function-based syntax is also required when the contraction pattern is not known at compile time but is rather determined dynamically.

## Simple mode

The basic exposed interface makes use of any iterable `IA`, `IB` or `IC` to denote labels of indices, in a similar fashion as when used in the context of [`@tensor`](@ref).
When making use of this functionality, in-place operations are no longer supported, as these are reserved for the *expert mode*.
Note that the return type is only inferred when the labels are entered as tuples, and also `IC` is specified.


```@docs
tensorcopy
tensoradd
tensortrace
tensorcontract
tensorproduct
```

## Expert mode

The expert mode exposes both mutating and non-mutating versions of these functions.
In this case, selected indices are determined through permutations, specified by `pA`, `pB` and `pC`.
In order to distinguish from the non-mutating version in *simple mode*, overlapping functionality is distinguished by specializing on these permutations, which are of type [`Index2Tuple`](@ref).

These functions come in a mutating and non-mutating version. The mutating versions mimic
the argument order of some of the BLAS functions, such as `blascopy!`, `axpy!` and `gemm!`.
Symbols `A` and `B` always refer to input arrays, whereas `C` is used to denote the array
where the result will be stored. They also return `C` and are therefore type stable. The
greek letters `α` and `β` denote scalar coefficients.

```@docs
tensorcopy!
tensoradd!
tensortrace!
tensorcontract!
tensorproduct!
```

The non-mutating functions are simpler in not allowing scalar coefficients and conjugation.
They also take a default value for the labels of the output array if these are not
specified. However, the return type is only inferred if the labels are entered as tuples,
and also `IC` is specified. They are simply called as:

```@docs
tensorcopy
tensoradd
tensortrace
tensorcontract
tensorproduct
```
