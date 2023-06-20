# Functions

The elementary tensor operations can also be accessed via functions, mainly for compatibility with older versions of this toolbox.
The function-based syntax is also required when the contraction pattern is not known at compile time but is rather determined dynamically.

The basic exposed interface makes use of any iterable `IA`, `IB` or `IC` to denote labels of indices, in a similar fashion as when used in the context of [`@tensor`](@ref).
When making use of this functionality, in-place operations are no longer supported, as these are reserved for the *expert mode*.
Note that the return type is only inferred when the labels are entered as tuples, and also `IC` is specified.

The expert mode exposes both mutating and non-mutating versions of these functions.
In this case, selected indices are determined through permutations, specified by `pA`, `pB` and `pC`.
In order to distinguish from the non-mutating version in *simple mode*, overlapping functionality is distinguished by specializing on these permutations, which are of type [`Index2Tuple`](@ref).

```@docs
tensorcopy
tensoradd
tensortrace
tensorcontract
tensorproduct
```

```@docs
tensoradd!
tensortrace!
tensorcontract!
```

<!-- tensorcopy! and tensorproduct!? -->
