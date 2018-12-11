# Functions

The elementary tensor operations can also be accessed via functions, mainly for
compatibility with older versions of this toolbox. The function-based syntax is also
required when the contraction pattern is not known at compile time but is rather determined
dynamically.

These functions come in a mutating and non-mutating version. The mutating versions mimick
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
