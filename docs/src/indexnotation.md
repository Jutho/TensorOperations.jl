# Index notation with `@tensor` macro

The prefered way to specify (a sequence of) tensor operations is by using the `@tensor`
macro, which accepts an
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format, a.k.a.
[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) (and in particular,
Einstein's summation convention).

 This can most easily be explained using a simple example:

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

In the second to last line, the result of the operation will be stored in the preallocated
array `D`, whereas the last line uses a different assignment operator `:=` in order to
define a new array `E` of the correct size. The contents of `D` and `E` will be equal.

Following Einstein's summation convention, the result is computed by first tracing/
contracting the 3rd and 5th index of array `A`. The resulting array will then be contracted
with array `B` by contracting its 2nd index with the last index of `B` and its last index
with the first index of `B`. The resulting array has three remaining indices, which
correspond to the indices `a` and `c` of array `A` and index `b` of array `B` (in that
order). To this, the array `C` (scaled with `α`) is added, where its first two indices will
be permuted to fit with the order `a,c,b`. The result will then be stored in array `D`,
which requires a second permutation to bring the indices in the requested order `a,b,c`.

In this example, the labels were specified by arbitrary letters or even longer names. Any
valid variable name is valid as a label. Note though that these labels are never
interpreted as existing Julia variables, but rather are converted into symbols by the
`@tensor` macro. This means, in particular, that the specific tensor operations defined by
the code inside the `@tensor` environment are completely specified at compile time.
Alternatively, one can also choose to specify the labels using literal integer constants,
such that also the following code specifies the same operation as above. Finally, it is
also allowed to use primes (i.e. Julia's `adjoint` operator) to denote different indices,
including using multiple subsequent primes.

```julia
@tensor D[å'',ß,c'] = A[å'',1,-3,c',-3,2]*B[2,ß,1] + α*C[c',å'',ß]
```

The index pattern is analyzed at compile time and expanded to a set of calls to the basic
tensor operations, i.e. [`add!`](@ref), [`trace!`](@ref) and [`contract!`](@ref).
Temporaries are created where necessary, but will by default be saved to a global cache, so
that they can be reused upon a next iteration or next call to the function in which the
`@tensor` call is used. When experimenting in the REPL where every tensor expression is
only used a single time, it might be better to use `disable_cache()`, though no real harm
comes from using the cache (except higher memory usage). By default, the cache is allowed to take up to 50% of the total machine memory, though this is fully configurable.

A contraction of several tensors `A[a,b,c,d,e]*B[b,e,f,g]*C[c,f,i,j]*...` is evaluted using pairwise contractions, using Julia's default left to right order, i.e. as
`( (A[a,b,c,d,e] * B[b,e,f,g]) * C[c,f,i,j]) * ...`. However, if one respects the so-called
[NCON](https://arxiv.org/abs/1402.0939) style of specifying indices, i.e. positive integers
for the contracted indices and negative indices for the open indices, the different factors
will be reordered and so that the pairwise tensor contractions contract over indices with
smaller integer label first. For example,
```julia
D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]
```
will be evaluated as `(A[-1,3,1,-2,2]*C[1,4,-4,-3])*B[3,2,4,-5]`. Furthermore, in that case
the indices of the output tensor (`D` in this case) do not need to be specified (using `[:]`
instead), and will be chosen as `(-1,-2,-3,-4,-5)`. Any other order is of course still
possible by just specifying it.

Furthermore, there is a `@tensoropt` macro which will optimize the contraction order to
minimize the total number of multiplications (cost model might change or become configurable
in the future). The optimal contraction order will be determined at compile time and will
be hard coded in the macro expansion. The cost/size of the different indices can be
specified in various ways, and can be integers or some arbitrary polynomial of an abstract
variable, e.g. `χ`. In the latter case, the optimization assumes the assymptotic limit of
large `χ`.

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
