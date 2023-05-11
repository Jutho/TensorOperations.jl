# methods/simple.jl
#
# Method-based access to tensor operations using simple definitions.

# type unstable; better use tuples instead
tensorcopy(A, IA, IC=IA) = tensorcopy(A, tuple(IA...), tuple(IC...))
tensorcopy(A, IA, CA::Symbol, IC=IA) = tensorcopy(A, tuple(IA...), CA, tuple(IC...))
tensoradd(A, IA, B, IB, IC=IA) = tensoradd(A, tuple(IA...), B, tuple(IB...), tuple(IC...))
tensoradd(A, IA, CA::Symbol, B, IB, CB::Symbol; IC=IA) = tensoradd(A, tuple(IA...), CA, B, tuple(IB...), CB, tuple(IC...))
tensortrace(A, IA, IC=unique2(IA)) = tensortrace(A, tuple(IA...), tuple(IC...))
tensortrace(A, IA, CA::Symbol, IC=unique2(IA)) = tensortrace(A, tuple(IA...), CA, tuple(IC...))
function tensorcontract(A, IA, B, IB, IC=symdiff(IA, IB))
    return tensorcontract(A, tuple(IA...), B, tuple(IB...), tuple(IC...))
end
function tensorcontract(A, IA, CA::Symbol, B, IB, CB::Symbol, IC=symdiff(IA, IB))
    return tensorcontract(A, tuple(IA...), CA, B, tuple(IB...), CB, tuple(IC...))
end
function tensorproduct(A, IA, B, IB, IC=vcat(IA, IB))
    return tensorproduct(A, tuple(IA...), B, tuple(IB...), tuple(IC...))
end
function tensorproduct(A, IA, CA::Symbol, B, IB, CB::Symbol, IC=vcat(IA, IB))
    return tensorproduct(A, tuple(IA...), CA, B, tuple(IB...), CB, tuple(IC...))
end

"""
    tensorcopy(A, IA, [CA], IC = IA)

Creates a copy of `A`, where the dimensions of `A` are assigned indices from the
iterable `IA` and the indices of the copy are contained in `IC`. Both iterables
should contain the same elements in a different order.

The result of this method is equivalent to `permutedims(A, p)` where p is the permutation
such that `IC = IA[p]`. The implementation of `tensorcopy` is however more efficient on
average, especially if `Threads.nthreads() > 1`.

Optionally, the symbol `CA` can be used to specify that the input array should be conjugated.
"""
function tensorcopy(A, IA::Tuple, CA::Symbol, IC::Tuple=IA)
    pA = add_indices(IA, IC)
    TC = scalartype(A)
    C = tensoralloc_add(TC, A, pA, CA)
    return tensoradd!(C, A, pA, CA, one(TC), zero(TC))
end
tensorcopy(A, IA::Tuple, IC::Tuple=IA) = tensorcopy(A, IA, :N, IC)

"""
    tensoradd(A, IA, [CA], B, IB, [CB], IC = IA)

Returns the result of adding arrays `A` and `B` where the iterables `IA` and `IB`
denote how the array data should be permuted in order to be added. More specifically,
the result of this method is equivalent to

```julia
tensorcopy(A, IA, IC) + tensorcopy(B, IB, IC)
```

but without creating the temporary permuted arrays.

Optionally, the symbols `CA` and `CB` can be used to specify that the input arrays should be conjugated.
"""
function tensoradd(A, IA::Tuple, CA::Symbol, B, IB::Tuple, CB::Symbol, IC::Tuple=IA)
    TC = promote_add(scalartype(A), scalartype(B))
    pA = add_indices(IA, IC)
    C = tensoralloc_add(TC, A, pA, CA)
    tensoradd!(C, A, pA, CA, one(TC), zero(TC))
    pB = add_indices(IB, IC)
    return tensoradd!(C, B, pB, CB, one(TC), one(TC))
end
tensoradd(A, IA::Tuple, B, IB::Tuple, IC::Tuple=IA) = tensoradd(A, IA, :N, B, IB, :N, IC)

"""
    tensortrace(A, IA, [CA], [IC])

Trace or contract pairs of indices of array `A`, by assigning them an identical
indices in the iterable `IA`. The untraced indices, which are assigned a unique index,
can be reordered according to the optional argument `IC`. The default value corresponds
to the order in which they appear. Note that only pairs of indices can be contracted,
so that every index in `IA` can appear only once (for an untraced index) or twice
(for an index in a contracted pair).

Optionally, the symbol `CA` can be used to specify that the input array should be conjugated.
"""
function tensortrace(A, IA::Tuple, CA::Symbol, IC::Tuple)
    pC, cindA1, cindA2 = trace_indices(IA, IC)
    TC = promote_contract(scalartype(A))
    C = tensoralloc_add(TC, A, pC, CA)
    return tensortrace!(C, pC, A, (cindA1, cindA2), CA, one(TC), zero(TC))
end
tensortrace(A, IA::Tuple, IC::Tuple) = tensortrace(A, IA, :N, IC)

"""
    tensorcontract(A, IA, [CA], B, IB, [CB], IC)

Contract indices of array `A` with corresponding indices in array `B` by assigning
them identical labels in the iterables `IA` and `IB`. The indices of the resulting
array correspond to the indices that only appear in either `IA` or `IB` and can be
ordered by specifying the optional argument `IC`. The default is to have all open
indices of array `A` followed by all open indices of array `B`. Note that inner
contractions of an array should be handled first with `tensortrace`, so that every
label can appear only once in `IA` or `IB` seperately, and once (for open
index) or twice (for contracted index) in the union of `IA` and `IB`.

The contraction can be performed by a native Julia algorithm without creating any
temporaries, or by first permuting the arrays such that the contraction becomes equivalent
to a matrix product, which is then performed by BLAS. The latter is typically faster for
large arrays. The choice of method is globally controlled by the methods
[`enable_blas()`](@ref) and [`disable_blas()`](@ref).

Optionally, the symbols `CA` and `CB` can be used to specify that the input arrays should be conjugated.
"""
function tensorcontract(A, IA::Tuple, CA::Symbol, B, IB::Tuple, CB::Symbol, IC::Tuple)
    pA, pB, pC = contract_indices(IA, IB, IC)
    TC = promote_contract(scalartype(A), scalartype(B))
    C = tensoralloc_contract(TC, pC, A, pA, CA, B, pB, CB)
    return tensorcontract!(C, pC, A, pA, CA, B, pB, CB, one(TC), zero(TC))
end
function tensorcontract(A, IA::Tuple, B, IB::Tuple, IC::Tuple)
    return tensorcontract(A, IA, :N, B, IB, :N, IC)
end

"""
    tensorproduct(A, IA, [CA], B, IB, [CB], IC = (IA..., IB...))

Computes the tensor product of two arrays `A` and `B`, i.e. returns a new array `C`
with `ndims(C) = ndims(A)+ndims(B)`. The indices of the output tensor are related to
those of the input tensors by the pattern specified by the indices. Essentially,
this is a special case of `tensorcontract` with no indices being contracted over.
This method checks whether the indices indeed specify a tensor product instead of
a genuine contraction.

Optionally, the symbols `CA` and `CB` can be used to specify that the input arrays should be conjugated.
"""
function tensorproduct(A, IA::Tuple, CA::Symbol, B, IB::Tuple, CB::Symbol,
                       IC::Tuple=(IA..., IB...))
    isempty(intersect(IA, IB)) || throw(IndexError("not a valid tensor product"))
    return tensorcontract(A, IA, CA, B, IB, CB, IC)
end
function tensorproduct(A, IA::Tuple, B, IB::Tuple, IC::Tuple=(IA..., IB...))
    return tensorproduct(A, IA, :N, B, IB, :N, IC)
end
