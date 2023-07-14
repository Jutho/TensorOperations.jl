# Implementation

TODO: read and fix some referencing issues


The macros [`@tensor`](@ref) and the related [`@tensoropt`](@ref) work as
parsers for indexed tensor expressions. They transform these into a sequence of
calls to the primitive tensor operations. This allows the support of custom
types that implement the [Interface](@ref). The actual implementation is
achieved through the use of `TensorParser`, which provides the general framework
to parse tensor expressions. The `@tensor` macro is then just a wrapper around
this, which configures the default behavior and handles keyword arguments of the
parser.

The `TensorParser` works by breaking down the parsing into three main phases.
First, a basic check of the supplied expression is performed, to ensure that it
is a valid tensor expression. Then, a number of preprocessing steps can be
performed, which are used to standardize expressions, allow for syntactic sugar
features, and can also be used as a hook for writing custom parsers. Then, the
actual processing of the contractions is performed, which rewrites the
expression into a set of binary rooted trees, which is then used to generate the
actual calls to the primitive tensor operations. Finally, a number of
postprocessing steps can be added, which are mostly used to clean up the
resulting expression by flattening and by removing line number nodes, but also
to incorporate the custom backend system.

## Verifiers

The basic checks are performed by [`verifytensorexpr`](@ref), which calls the verifiers
[`isassignment`](@ref), [`isdefinition`](@ref), [`istensor`](@ref), [`istensorexpr`](@ref)
and [`isscalarexpr`](@ref).

```@docs
TensorOperations.verifytensorexpr
TensorOperations.isassignment
TensorOperations.isdefinition
TensorOperations.isindex
TensorOperations.istensor
TensorOperations.istensorexpr
TensorOperations.isscalarexpr
```

## Preprocessing

```@docs
TensorOperations.normalizeindices
TensorOperations.expandconj
TensorOperations.groupscalarfactors
TensorOperations.nconindexcompletion
TensorOperations.extracttensorobjects
TensorOperations.insertcontractionchecks
```

## Processing

```@docs
TensorOperations.processcontractions
TensorOperations.tensorify
```

## Postprocessing

```@docs
TensorOperations._flatten
TensorOperations.removelinenumbernode
TensorOperations.addtensoroperations
TensorOperations.insertbackend
```

## Analysis of contraction graphs and optimizing contraction order

The macro [`@tensoropt`](@ref) or the combination of [`@tensor`](@ref) with the keyword
`opt` can be used to optimize the contraction order of the expression at compile time. This
is done by analyzing the contraction graph, where the nodes are the tensors and the edges
are the contractions, in combination with the data provided in `optdata`, which is a
dictionary associating a cost (either a number or a polynomial in some abstract scaling
parameter) to every index. This information is then used to determine the (asymptotically)
optimal contraction tree (in terms of number of floating point operations). The algorithm
that is used is described in [arXiv:1304.6112](https://arxiv.org/abs/1304.6112).
