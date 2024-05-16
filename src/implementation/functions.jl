# methods/simple.jl
#
# Method-based access to tensor operations using simple definitions.

# ------------------------------------------------------------------------------------------
# tensorcopy
# ------------------------------------------------------------------------------------------

"""
    tensorcopy([IC=IA], A, IA, [conjA=:N, [α=1]])
    tensorcopy(A, pA::Index2Tuple, conjA, α) # expert mode

Create a copy of `A`, where the dimensions of `A` are assigned indices from the
iterable `IA` and the indices of the copy are contained in `IC`. Both iterables
should contain the same elements, optionally in a different order.

The result of this method is equivalent to `α * permutedims(A, pC)` where `pC` is the
permutation such that `IC = IA[pC]`. The implementation of `tensorcopy` is however more
efficient on average, especially if `Threads.nthreads() > 1`.

Optionally, the symbol `conjA` can be used to specify whether the input tensor should be
conjugated (`:C`) or not (`:N`).

See also [`tensorcopy!`](@ref).
"""
function tensorcopy end

function tensorcopy(IC::Tuple, A, IA::Tuple, conjA::Symbol=:N, α::Number=One())
    pA = add_indices(IA, IC)
    return tensorcopy(A, pA, conjA, α)
end
# default `IC`
function tensorcopy(A, IA, conjA::Symbol=:N, α::Number=One())
    return tensorcopy(tuple(IA...), A, tuple(IA...), conjA, α)
end
# implement for iterables
function tensorcopy(IC, A, IA, conjA::Symbol=:N, α::Number=One())
    return tensorcopy(tuple(IC...), A, tuple(IA...), conjA, α)
end
# expert mode
function tensorcopy(A, pA::Index2Tuple, conjA::Symbol=:N, α::Number=One(),
                    backend::Backend...)
    TC = promote_add(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, A, pA, conjA)
    return tensorcopy!(C, A, pA, conjA, α, backend...)
end

"""
    tensorcopy!(C, A, pA::Index2Tuple, conjA=:N, α=1, [backend])

Copy the contents of tensor `A` into `C`, where the dimensions `A` are permuted according to
the permutation and repartition `pA`.

The result of this method is equivalent to `α * permutedims!(C, A, pA)`.

Optionally, the symbol `conjA` can be used to specify whether the input tensor should be
conjugated (`:C`) or not (`:N`).

!!! warning 
    The object `C` must not be aliased with `A`.

See also [`tensorcopy`](@ref) and [`tensoradd!`](@ref)
"""
function tensorcopy!(C, A, pA::Index2Tuple, conjA::Symbol=:N, α::Number=One(),
                     backend::Backend...)
    return tensoradd!(C, A, pA, conjA, α, false, backend...)
end

# ------------------------------------------------------------------------------------------
# tensoradd
# ------------------------------------------------------------------------------------------

"""
    tensoradd([IC=IA], A, IA, [conjA], B, IB, [conjB], [α=1, [β=1]])
    tensoradd(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, pB::Index2Tuple, conjB, α=1, β=1, [backend]) # expert mode

Return the result of adding arrays `A` and `B` where the iterables `IA` and `IB`
denote how the array data should be permuted in order to be added. More specifically,
the result of this method is equivalent to
`α * permutedims(A, pA) + β * permutedims(B, pB)` where `pA` (`pB`) is the permutation such
that `IC = IA[pA]` (`IB[pB]`). The implementation of `tensoradd` is however more efficient
on average, as the temporary permuted arrays are not created.

Optionally, the symbols `conjA` and `conjB` can be used to specify whether the input tensors
should be conjugated (`:C`) or not (`:N`).

See also [`tensoradd!`](@ref).
"""
function tensoradd end

function tensoradd(IC::Tuple, A, IA::Tuple, conjA::Symbol, B, IB::Tuple,
                   conjB::Symbol, α::Number=One(), β::Number=One())
    return tensoradd(A, add_indices(IA, IC), conjA, B, add_indices(IB, IC), conjB, α, β)
end
# default `IC`
function tensoradd(A, IA, conjA::Symbol, B, IB, conjB::Symbol,
                   α::Number=One(), β::Number=One())
    return tensoradd(tuple(IA...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α, β)
end
# default `conjA` and `conjB`
function tensoradd(IC, A, IA, B, IB, α::Number=One(), β::Number=One())
    return tensoradd(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α, β)
end
# default `IC`, `conjA` and `conjB`
function tensoradd(A, IA, B, IB, α::Number=One(), β::Number=One())
    return tensoradd(tuple(IA...), A, tuple(IA...), B, tuple(IB...), α, β)
end
# iterables
function tensoradd(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol,
                   α::Number=One(), β::Number=One())
    return tensoradd(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α, β)
end
# expert mode
function tensoradd(A, pA::Index2Tuple, conjA::Symbol,
                   B, pB::Index2Tuple, conjB::Symbol,
                   α::Number=One(), β::Number=One(), backend::Backend...)
    TC = promote_add(scalartype(A), scalartype(B), scalartype(α), scalartype(β))
    C = tensoralloc_add(TC, A, pA, conjA)
    C = tensorcopy!(C, A, pA, conjA, α)
    return tensoradd!(C, B, pB, conjB, β, One(), backend...)
end

# ------------------------------------------------------------------------------------------
# tensortrace
# ------------------------------------------------------------------------------------------

"""
    tensortrace([IC], A, IA, [conjA], [α=1])
    tensortrace(A, p::Index2Tuple, q::Index2Tuple, conjA, α=1, [backend]) # expert mode

Trace or contract pairs of indices of tensor `A`, by assigning them identical indices in the
iterable `IA`. The untraced indices, which are assigned a unique index, can be reordered
according to the optional argument `IC`. The default value corresponds to the order in which
they appear. Note that only pairs of indices can be contracted, so that every index in `IA`
can appear only once (for an untraced index) or twice (for an index in a contracted pair).

Optionally, the symbol `conjA` can be used to specify that the input tensor should be
conjugated.

See also [`tensortrace!`](@ref).
"""
function tensortrace end

# default `IC`
function tensortrace(A, IA, conjA::Symbol, α::Number=One())
    return tensortrace(unique2(tuple(IA...)), A, tuple(IA...), conjA, α)
end
# default `conjA`
function tensortrace(IC, A, IA, α::Number=One())
    return tensortrace(tuple(IC...), A, tuple(IA...), :N, α)
end
# default `IC` and `conjA`
function tensortrace(A, IA, α::Number=One())
    return tensortrace(unique2(tuple(IA...)), A, tuple(IA...), :N, α)
end
# labels to indices
function tensortrace(IC, A, IA, conjA::Symbol, α::Number=One())
    p, q = trace_indices(tuple(IA...), tuple(IC...))
    return tensortrace(A, p, q, conjA, α)
end
# expert mode
function tensortrace(A, p::Index2Tuple, q::Index2Tuple, conjA::Symbol, α::Number=One(),
                     backend::Backend...)
    TC = promote_contract(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, A, p, conjA)
    return tensortrace!(C, A, p, q, conjA, α, Zero(), backend...)
end

# ------------------------------------------------------------------------------------------
# tensorcontract
# ------------------------------------------------------------------------------------------

"""
    tensorcontract([IC], A, IA, [conjA], B, IB, [conjB], [α=1])
    tensorcontract(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, [backend]) # expert mode

Contract indices of tensor `A` with corresponding indices in tensor `B` by assigning
them identical labels in the iterables `IA` and `IB`. The indices of the resulting
tensor correspond to the indices that only appear in either `IA` or `IB` and can be
ordered by specifying the optional argument `IC`. The default is to have all open
indices of `A` followed by all open indices of `B`. Note that inner contractions of an array
should be handled first with `tensortrace`, so that every label can appear only once in `IA`
or `IB` seperately, and once (for an open index) or twice (for a contracted index) in the
union of `IA` and `IB`.

Optionally, the symbols `conjA` and `conjB` can be used to specify that the input tensors
should be conjugated.

See also [`tensorcontract!`](@ref).
"""
function tensorcontract end

function tensorcontract(IC::Tuple, A, IA::Tuple, conjA::Symbol, B, IB::Tuple, conjB::Symbol,
                        α::Number=One())
    pA, pB, pAB = contract_indices(IA, IB, IC)
    return tensorcontract(A, pA, conjA, B, pB, conjB, pAB, α)
end
# default `IC`
function tensorcontract(A, IA, conjA, B, IB, conjB, α::Number=One())
    return tensorcontract(symdiff(tuple(IA...), tuple(IB...)), A, tuple(IA...), conjA, B,
                          tuple(IB...), conjB, α)
end
# default `conjA` and `conjB`
function tensorcontract(IC, A, IA, B, IB, α::Number=One())
    return tensorcontract(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α)
end
# default `IC`, `conjA` and `conjB`
function tensorcontract(A, IA, B, IB, α::Number=One())
    return tensorcontract(symdiff(tuple(IA...), tuple(IB...)), A, tuple(IA...), :N, B,
                          tuple(IB...), :N, α)
end
# iterables
function tensorcontract(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=One())
    return tensorcontract(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α)
end
# expert mode
function tensorcontract(A, pA::Index2Tuple, conjA::Symbol,
                        B, pB::Index2Tuple, conjB::Symbol,
                        pAB::Index2Tuple, α::Number=One(), backend::Backend...)
    TC = promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB)
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, Zero(), backend...)
end

# ------------------------------------------------------------------------------------------
# tensorproduct
# ------------------------------------------------------------------------------------------

"""
    tensorproduct([IC], A, IA, [conjA], B, IB, [conjB], [α=1])
    tensorproduct(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, [backend]) # expert mode

Compute the tensor product (outer product) of two tensors `A` and `B`, i.e. returns a new
tensor `C` with `ndims(C) = ndims(A) + ndims(B)`. The indices of the output tensor are
related to those of the input tensors by the pattern specified by the indices. Essentially,
this is a special case of [`tensorcontract`](@ref) with no indices being contracted over.
This method checks whether the indices indeed specify a tensor product instead of a genuine
contraction.

Optionally, the symbols `conjA` and `conjB` can be used to specify that the input tensors
should be conjugated.

See also [`tensorproduct!`](@ref) and [`tensorcontract`](@ref).
"""
function tensorproduct end

function tensorproduct(IC::Tuple, A, IA::Tuple, conjA::Symbol, B, IB::Tuple, conjB::Symbol,
                       α::Number=One())
    pA, pB, pAB = contract_indices(IA, IB, IC)
    return tensorproduct(A, pA, conjA, B, pB, conjB, pAB, α)
end
# default `IC`
function tensorproduct(A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=One())
    return tensorproduct(vcat(tuple(IA...), tuple(IB...)), A, tuple(IA...), conjA, B,
                         tuple(IB...), conjB, α)
end
# default `conjA` and `conjB`
function tensorproduct(IC, A, IA, B, IB, α::Number=One())
    return tensorproduct(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α)
end
# default `IC`, `conjA` and `conjB`
function tensorproduct(A, IA, B, IB, α::Number=One())
    return tensorproduct(vcat(tuple(IA...), tuple(IB...)), A, tuple(IA...), :N, B,
                         tuple(IB...), :N, α)
end
# iterables
function tensorproduct(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=One())
    return tensorproduct(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α)
end
# expert mode
function tensorproduct(A, pA::Index2Tuple, conjA::Symbol,
                       B, pB::Index2Tuple, conjB::Symbol,
                       pAB::Index2Tuple, α::Number=One(), backend::Backend...)
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract(A, pA, conjA, B, pB, conjB, pAB, α, backend...)
end

"""
    tensorproduct!(C, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, β=0)

Compute the tensor product (outer product) of two tensors `A` and `B`, i.e. a wrapper of
[`tensorcontract!`](@ref) with no indices being contracted over. This method checks whether
the indices indeed specify a tensor product instead of a genuine contraction.

!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See als [`tensorproduct`](@ref) and [`tensorcontract!`](@ref).
"""
function tensorproduct!(C,
                        A, pA::Index2Tuple, conjA::Symbol,
                        B, pB::Index2Tuple, conjB::Symbol,
                        pAB::Index2Tuple,
                        α::Number=One(), β::Number=Zero(), backend::Backend...)
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend...)
end
