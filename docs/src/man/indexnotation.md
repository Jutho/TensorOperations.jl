# Index notation with macros

The main export and main functionality of TensorOperations.jl is the `@tensor` macro

```@docs
@tensor
```

The functionality and configurability of `@tensor` and some of its relatives is explained in
detail on this page.

## The `@tensor` macro

The prefered way to specify (a sequence of) tensor operations is by using the `@tensor`
macro, which accepts an
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format, a.k.a.
[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) (and in particular,
Einstein's summation convention).

This can most easily be explained using a simple example:

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

The first thing to note is the use of two different assignment operators within the body of
the `@tensor` call. The regular assignment operator `=` stores the result of the tensor
expression in the right hand side in an existing tensor `D`, whereas the alternative
assignment operator `:=` results in a new tensor `E` with the correct properties to be
created. However, the contents of `D` and `E` will be equal.

Following Einstein's summation convention, these contents are computed in a number of steps
involving the three primitive tensor operators. In this particular example, the first step
involves tracing/ contracting the 3rd and 5th index of array `A`. The resulting array will
then be contracted with array `B` by contracting its 2nd index with the last index of `B`
and its last index with the first index of `B`. The resulting array has three remaining
indices, which correspond to the indices `a` and `c` of array `A` and index `b` of array `B`
(in that order). To this, the array `C` (scaled with `α`) is added, where its first two
indices will be permuted to fit with the order `a, c, b`. The result will then be stored in
array `D` (or `E`), which requires a second permutation to bring the indices in the
requested order `a, b, c`.

The index pattern is analyzed at compile time and expanded to a set of calls to the basic
tensor operations, i.e. [`tensoradd!`](@ref), [`tensortrace!`](@ref) and
[`tensorcontract!`](@ref). Temporaries are created where necessary, as these building blocks
operate pairwise on the input tensors. The generated code can easily be inspected

```@example
using TensorOperations
@macroexpand @tensor E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
```
The different functions in which this tensor expression is decomposed are discussed in more
detail in the [Implementation](@ref) section of this manual.

In this example, the tensor indices were labeled with arbitrary letters; also longer names
could have been used. In fact, any proper Julia variable name constitutes a valid label.
Note though that these labels are never interpreted as existing Julia variables. Within the
`@tensor` macro they are converted into symbols and then used as dummy names, whose only
role is to distinguish the different indices. Their specific value bears no meaning. They
also do not appear in the generated code as illustrated above. This implies, in particular,
that the specific tensor operations defined by the code inside the `@tensor` environment are
completely specified at compile time. Various remarks regarding the index notation are in
order.

1.  TensorOperations.jl only supports strict Einstein summation convention. This implies
    that there are two types of indices. Either an index label appears once in every term of
    the right hand side, and it also appears on the left hand side. We refer to the
    corresponding indices as *open or free*. Alternatively, an index label appears exactly
    twice within a given term on the right hand side. The corresponding indices are referred
    to as *closed or contracted*, i.e. the pair of indices takes equal values and are summed
    over their (equal) range. This is known as a contraction, either an outer contraction
    (between two indices of two different tensors) or an inner contraction (a.k.a. trace,
    between two indices of a single tensor). More liberal use of the index notation, such as
    simultaneous summutation over three or more indices, or a open index appearing
    simultaneously in different tensor factors, are not supported by TensorOperations.jl

2.  Aside from valid Julia identifiers, index labels can also be specified using literal
    integer constants or using a combination of integers and symbols. Furthermore, it is
    also allowed to use primes (i.e. Julia's `adjoint` operator) to denote different
    indices, including using multiple subsequent primes. The following expression thus
    computes the same result as the example above:

    ```julia
    @tensor D[å'', ß, clap'] = A[å'', 1, -3, clap', -3, 2] * B[2, ß, 1] + α * C[c', å'', ß]
    ```

3.  If only integers are used for specifying index labels, this can be used to control the
    pairwise contraction order, by using the well-known NCON convention, where 'open
    indices' (appearing) in the left hand side are labelled by negative integers `-1`, `-2`,
    `-3`, whereas contraction indices are labelled with positive integers `1`, `2`, … Since
    the index order of the left hand side is in that case clear from the right hand side
    expression, the left hand side can be indexed with `[:]`, which is automatically
    replaced with all negative integers appearing in the right hand side, in decreasing
    order. The value of the labels for the contraction indices determines the pairwise
    contraction order. If multiple tensors need to be contracted, a first temporary will be
    created consisting of the contraction of the pair of tensors that share contraction
    index `1`, then the pair of tensors that share contraction index `2` (if not contracted
    away in the first pair) will be contracted, and so forth. The next subsection explains
    contraction order in more detail and gives some useful examples, as the example above
    only includes a single pair of tensors to be contracted.

4.  Index expressions `[...]` are only interpreted as index notation on the highest level.
    For example, if you want to mulitply two matrices which are stored in a list, you can
    write
    
    ```
    @tensor result[i,j] := list[1][i,k] * list[2][k,j]
    ```

    However, if both are stored as a the slices of a 3-way array, you cannot write

    ```
    @tensor result[i,j] := list[i,k,1] * list[k,j,2]
    ```

    Rather, you should use

    ```
    @tensor result[i,j] := list[:,:,1][i,k] * list[:,:,2][k,j]
    ```

    or, if you want to avoid additional allocations

    ```
    @tensor result[i,j] := view(list, :,:,1)[i,k] * view(list, :,:,2)[k,j]
    ```

Note, finally, that the `@tensor` specifier can be put in front of a single tensor
expression, or in front of a `begin ... end` block to group and evaluate different
expressions at once. Within an `@tensor begin ... end` block, the `@notensor` macro can be
used to annotate indexing expressions that need to be interpreted literally. The previous
matrix multiplication example with matrices stored in a 3-way array could for example also
be written as

```julia
@tensor begin
    @notensor A = list[:,:,1]
    @notensor B = list[:,:,2]
    result[i,j] = A[i,k] * B[k,j]
end
```

## Contraction order specification and optimisation

A contraction of several (more than two) tensors, as in

```julia
@tensor D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
```

is known as a *tensor network* and is generically evaluated as a sequence of pairwise
contractions. In the example above, this contraction is evaluated using Julia's default left
to right order, i.e. as `(A[a, b, c, d, e] * B[b, e, f, g]) * C[c, f, i, j]`. There are
however different strategies to modify this order.

1.  Explicit parenthesis can be used to group subnetworks within a tensor network that will
    be evaluated first. Parentheses around subexpressions are thus always respected by the
    `@tensor` macro.

2.  As explained in the previous subsection, if one respects the
    [NCON](https://arxiv.org/abs/1402.0939) convention of specifying indices, i.e. positive
    integers for the contracted indices and negative indices for the open indices, the
    different factors will be reordered and so that the pairwise tensor contractions
    contract over indices with smaller integer label first. For example,

    ```
    @tensor D[:] := A[-1, 3, 1, -2, 2] * B[3, 2, 4, -5] * C[1, 4, -4, -3]
    ```

    will be evaluated as `(A[-1, 3, 1, -2, 2] * C[1, 4, -4, -3]) * B[3, 2, 4, -5]`.
    Furthermore, in that case the indices of the output tensor (`D` in this case) do not
    need to be specified (using `[:]` instead), and will be chosen as
    `(-1, -2, -3, -4, -5)`. Note that if two tensors are contracted, all contraction indices
    among them will be contracted, even if there are additional contraction indices whose
    label is a higher positive number. For example,

    ```
    @tensor D[:] := A[-1, 3, 2, -2, 1] * B[3, 1, 4, -5] * C[2, 4, -4, -3]
    ```

    amounts to the original left to right order, because `A` and `B` share the first
    contraction index `1`. When `A` and `B` are contracted, also the contraction with label
    `3` will be performed, even though contraction index with label `2` is not yet
    'processed'.

3.  A specific contraction order can be manually specified by supplying an `order` keyword
    argument to the `@tensor` macro. The value is a tuple of the contraction indices in the
    order that they should be dealt with, e.g. the default order could be changed to first
    contract `A` with `C` using
    
    ```
    @tensor order=(c,b,e,f) begin
        D[a, d, j, i, g] := A[a, b, c, d, e] * B[b, e, f, g] * C[c, f, i, j]
    end
    ```

    Here, the same comment as in the NCON style applies; once two tensors are contracted
    because they share an index label which is next in the `order` list, all other indices
    with shared label among them will be contracted, irrespective of their order.

In the case of more complex tensor networks, the optimal contraction order cannot always
easily be guessed or determined on plain sight. It is then useful to be able to optimise the
contraction order automatically, given a model for the complexity of contracting the
different tensors in a particular order. This functionality is provided where the cost
function being minimised models the computational complexity by counting the number of
scalar multiplications. This minimisation problem is solved using the algorithm that was
described in
[Physical Review E 90, 033315 (2014)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.033315).
For a given tensor networ contraction, this algorithm is ran once at compile time, while
lowering the tensor espression, and the outcome will be hard coded in the expression
resulting from the macro expansion. While the computational complexity of this optimisation
algorithm scales itself exponentially in the number of tensors involved in the network, it
should still be acceptibly fast (milliseconds up to a few seconds at most) for tensor
network contractions with up to around 30 tensors. Information of the optimization process
can be obtained during compilation by using the alternative macro `@tensoropt_verbose`.

As the cost is determined at compile time, it is not using actual tensor properties (e.g.
`size(A, i)` in the case of arrays) in the cost model, and the cost or extent associated
with every index can be specified in various ways, either using integers or floating point
numbers or some arbitrary univariate polynomial of an abstract variable, e.g. `χ`. In the
latter case, the optimization assumes the asymptotic limit of large `χ`.

```julia
@tensoropt D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
@tensor opt=true D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
# cost χ for all indices (a, b, c, d, e, f)

@tensoropt (a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
@tensor opt=(a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
# cost χ for indices (a, b, c, e), other indices (d, f) have cost 1

@tensoropt !(a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
@tensor opt=!(a, b, c, e) D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
# cost 1 for indices (a, b, c, e), other indices (d, f) have cost χ

@tensoropt (a => χ, b => χ^2, c => 2 * χ, e => 5) begin
    D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
end
@tensor opt=(a => χ, b => χ^2, c => 2 * χ, e => 5) begin
    D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
end
# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
```

## Dynamical tensor network contractions with `ncon` and `@ncon`

Tensor network practicioners are probably more familiar with the network contractor function
`ncon` to perform a tensor network contraction, as e.g. described in
[NCON](https://arxiv.org/abs/1402.0939). In particular, a graphical application
[TensorTrace](https://www.tensortrace.com) was recently introduced to facilitate the
generation of such `ncon` calls. TensorOperations.jl provides compatibility with this
interface by also exposing an `ncon` function with the same basic syntax

```julia
ncon(list_of_tensor_objects, list_of_index_lists)
```

e.g. the example of above is equivalent to

```@example
@tensor D[:] := A[-1, 3, 1, -2, 2] * B[3, 2, 4, -5] * C[1, 4, -4, -3]
D ≈ ncon((A, B, C), ([-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]))
```

where the lists of tensor objects and of index lists can be given as a vector or a tuple.
The `ncon` function necessarily needs to analyze the contraction pattern at runtime, but
this can be an advantage, in cases where the contraction is determined by runtime
information and thus not known at compile time. A downside from this, besides the fact that
this can result in some overhead (though this is typically negligable for anything but very
small tensor contractions), is that `ncon` is type-unstable, i.e. its return type cannot be
inferred by the Julia compiler.

The full call syntax of the `ncon` method exposed by TensorOperations.jl is

```julia
ncon(tensorlist, indexlist, [conjlist]; order=..., output=...)
```

where the first two arguments are those of above. Let us first discuss the keyword
arguments. The keyword argument `order` can be used to change the contraction order, i.e. by
specifying which contraction indices need to be processed first, rather than the strictly
increasing order `[1, 2, ...]`, as discussed in the previous subsection. The keyword
argument `output` can be used to specify the order of the output indices, when it is
different from the default `[-1, -2, ...]`.

The optional positional argument `conjlist` is a list of `Bool` variables that indicate
whether the corresponding tensor needs to be conjugated in the contraction. So while

```julia
ncon([A, conj(B), C], [[-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]]) ≈
ncon([A, B, C], [[-1, 3, 1, -2, 2], [3, 2, 4, -5], [1, 4, -4, -3]], [false, true, false])
```

the latter has the advantage that conjugating `B` is not an extra step (which creates an
additional temporary requiring allocations), but is performed at the same time when it is
contracted.

As an alternative solution to the optional positional arguments, there is also an `@ncon`
macro. It is just a simple wrapper over an `ncon` call and thus does not analyze the
indices at compile time, so that they can be fully dynamical. However, it will transform

```julia
@ncon([A, conj(B), C], indexlist; order=..., output=...)
```

into

```julia
ncon(Any[A, B, C], indexlist, [false, true, false]; order=..., output=...)
```

so as to get the advantages of just-in-time conjugation (pun intended) using the familiar
looking `ncon` syntax.

As a proof of principle, let us study the following method for computing the environment to
the `W` isometry in a MERA, as taken from [Tensors.net](https://www.tensors.net/mera),
implemented in three different ways:

```julia
function IsoEnvW1(hamAB, hamBA, rhoBA, rhoAB, w, v, u)
    indList1 = Any[[7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11], [8, 10, 5, 6],
                   [1, 11, 2], [1, 6, 3]]
    indList2 = Any[[1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8], [1, 2, 11, 9],
                   [5, 8, 6], [5, 9, 7]]
    indList3 = Any[[5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2], [4, 5, 11, 6],
                   [1, 2, 8], [7, 6, 9]]
    indList4 = Any[[3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]]
    wEnv = ncon(Any[hamAB, rhoBA, conj(w), u, conj(u), v, conj(v)], indList1) +
           ncon(Any[hamBA, rhoBA, conj(w), u, conj(u), v, conj(v)], indList2) +
           ncon(Any[hamAB, rhoBA, conj(w), u, conj(u), v, conj(v)], indList3) +
           ncon(Any[hamBA, rhoAB, v, conj(v), conj(w)], indList4)
    return wEnv
end

function IsoEnvW2(hamAB, hamBA, rhoBA, rhoAB, w, v, u)
    indList1 = Any[[7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11], [8, 10, 5, 6],
                   [1, 11, 2], [1, 6, 3]]
    indList2 = Any[[1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8], [1, 2, 11, 9],
                   [5, 8, 6], [5, 9, 7]]
    indList3 = Any[[5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2], [4, 5, 11, 6],
                   [1, 2, 8], [7, 6, 9]]
    indList4 = Any[[3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]]
    wEnv = @ncon(Any[hamAB, rhoBA, conj(w), u, conj(u), v, conj(v)], indList1) +
           @ncon(Any[hamBA, rhoBA, conj(w), u, conj(u), v, conj(v)], indList2) +
           @ncon(Any[hamAB, rhoBA, conj(w), u, conj(u), v, conj(v)], indList3) +
           @ncon(Any[hamBA, rhoAB, v, conj(v), conj(w)], indList4)
    return wEnv
end

function IsoEnvW3(hamAB, hamBA, rhoBA, rhoAB, w, v, u)
    @tensor wEnv[-1, -2, -3] := hamAB[7, 8, -1, 9] * rhoBA[4, 3, -3, 2] * conj(w[7, 5, 4]) *
                                u[9, 10, -2, 11] * conj(u[8, 10, 5, 6]) * v[1, 11, 2] *
                                conj(v[1, 6, 3]) +
                                hamBA[1, 2, 3, 4] * rhoBA[10, 7, -3, 6] * conj(w[-1, 11, 10]) *
                                u[3, 4, -2, 8] * conj(u[1, 2, 11, 9]) * v[5, 8, 6] *
                                conj(v[5, 9, 7]) +
                                hamAB[5, 7, 3, 1] * rhoBA[10, 9, -3, 8] * conj(w[-1, 11, 10]) *
                                u[4, 3, -2, 2] * conj(u[4, 5, 11, 6]) * v[1, 2, 8] *
                                conj(v[7, 6, 9]) +
                                hamBA[3, 7, 2, -1] * rhoAB[5, 6, 4, -3] * v[2, 1, 4] *
                                conj(v[3, 1, 5]) * conj(w[7, -2, 6])
    return wEnv
end
```

All indices appearing in this problem are of size `χ`. For tensors with `ComplexF64` eltype
and values of `χ` in `2:2:32`, the reported minimal times using the `@belapsed` macro from
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) are given by

| χ  | IsoEnvW1: ncon | IsoEnvW2: @ncon | IsoEnvW3: @tensor |
|:-- |:-------------- |:--------------- |:----------------- |
| 2  | 0.000154413    | 0.000348091     | 6.4897e-5         |
| 4  | 0.000208224    | 0.000400065     | 9.5601e-5         |
| 6  | 0.000558442    | 0.00076453      | 0.000354621       |
| 8  | 0.00138887     | 0.00150175      | 0.000982109       |
| 10 | 0.00506386     | 0.00365188      | 0.00288137        |
| 12 | 0.0126571      | 0.00959403      | 0.00818371        |
| 14 | 0.0292822      | 0.0216231       | 0.0184712         |
| 16 | 0.0531353      | 0.0410914       | 0.0359749         |
| 18 | 0.225333       | 0.0774705       | 0.0688475         |
| 20 | 0.43358        | 0.139873        | 0.129315          |
| 22 | 0.601685       | 0.243468        | 0.221995          |
| 24 | 0.902662       | 0.459746        | 0.427615          |
| 26 | 1.2379         | 0.66722         | 0.622856          |
| 28 | 1.84234        | 1.08766         | 1.0322            |
| 30 | 2.58548        | 1.53826         | 1.44854           |
| 32 | 3.85758        | 2.44087         | 2.34229           |

Throughout this range of `χ` values, method 3 that uses the `@tensor` macro is consistenly
the fastest, both at small `χ`, where the type stability and the fact that the contraction
pattern is analyzed at compile time matters, and at large `χ`, where the caching of
temporaries matters. The direct `ncon` call has neither of those two features (unless the
fourth positional argument is specified, which was not the case here). The `@ncon` solution
provides a hook into the cache and thus is competitive with `@tensor` for large `χ`, where
the cost is dominated by matrix multiplication and allocations. For small `χ`, `@ncon` is
also plagued by the runtime analysis of the contraction, but is even worse then
`ncon`. For small `χ`, the unavoidable type instabilities in `ncon` implementation seem to
make the interaction with the cache hurtful rather than advantageous.

## Index compatibility and checks

Indices with the same label, either open indices on the two sides of the equation, or
contracted indices, need to be compatible. For `AbstractArray` objects, this means they must
have the same size. Other tensor types might have more complicated structure associated with
their indices, and requires matching between those. The function
[`checkcontractible`](@ref) is part of the interface that can be used to control when
tensors can be contracted with each other along specific indices.

If indices don't match, the contraction will spawn an error. However, this can be an error
deep within the implementation, at which point the error message will provide little
information as to which specific tensors and which indices are producing the mismatch. By
adding the keyword argument `contractcheck = true` to the `@tensor` macro, explicit checks
are enabled that are run before any tensor operation is performed, and when a mismatch is
detected, the still have the label information to spawn a more useful error message.


TODO: continue the following
A different type of check is the `costcheck` keyword argument, which can be given the values
`:warn` or `:cache`.

## Backends, multithreading and GPUs

Every index expression will be evaluated as a sequence of elementary tensor operations, i.e.
permuted additions, partial traces and contractions, which are implemented for strided
arrays as discussed in [Package features](@ref). In particular, these implementations rely
on [Strided.jl](https://github.com/Jutho/Strided.jl), and we refer to this package for a
full specification of which arrays are supported. As a rule of thumb, this primarily
includes `Array`s from Julia base, as well as `view`s thereof if sliced with a combination
of `Integer`s and `Range`s. Special types such as `Adjoint` and `Transpose` from Base are
also supported. For permuted addition and partial traces, native Julia implementations are
used which could benefit from multithreading if `JULIA_NUM_THREADS>1`. The binary
contraction is performed by first permuting the two input tensors into a form such that the
contraction becomes equivalent to one matrix multiplication on the whole data, followed by a
final permutation to bring the indices of the output tensor into the desired order. This
approach allows to use the highly efficient matrix multiplication kernel (`gemm`) from BLAS,
which is multithreaded by default. There is also a native contraction implementation that is
used for e.g. arrays with an `eltype` that is not `<:LinearAlgebra.BlasFloat`. It performs
the contraction directly without the additional permutations, but still in a cache-friendly
and multithreaded way (again relying on `JULIA_NUM_THREADS>1`). This implementation can also
be used for `BlasFloat` types (but will typically be slower), and the use of BLAS can be
controlled by explicitly switching the backend between `StridedBLAS` and `StridedNative`.

The primitive tensor operations are also implemented for `CuArray` objects of the
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) library. This implementation is essentially a
simple wrapper over the cuTENSOR library of NVidia, and will only be loaded when the
`cuTENSOR` library is loaded. The `@tensor` macro will then automatically work for
operations between GPU arrays.

Mixed operations between host arrays (e.g. `Array`) and device arrays (e.g. `CuArray`) will
fail however. If one wants to harness the computing power of the GPU to perform all tensor
operations, there is a dedicated macro `@cutensor`. This will transfer all host arrays to
the GPU before performing the requested operations. If the output is an existing host array,
the result will be copied back. If a new result array is created (i.e. using `:=`), it will
remain on the GPU device and it is up to the user to transfer it back. Arrays are transfered
to the GPU just before they are first used, and in a complicated tensor expression, this
might have the benefit that transer of the later arrays overlaps with computation of earlier
operations.
