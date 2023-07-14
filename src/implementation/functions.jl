# methods/simple.jl
#
# Method-based access to tensor operations using simple definitions.

# ------------------------------------------------------------------------------------------
# tensorcopy
# ------------------------------------------------------------------------------------------

"""
    tensorcopy([IC=IA], A, IA, [conjA=:N, [α=true]])
    tensorcopy(pC::Index2Tuple, A, conjA, α) # expert mode

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

function tensorcopy(IC::Tuple, A, IA::Tuple, conjA::Symbol=:N, α::Number=true)
    pC = add_indices(IA, IC)
    return tensorcopy(pC, A, conjA, α)
end
# default `IC`
function tensorcopy(A, IA, conjA::Symbol=:N, α::Number=true)
    return tensorcopy(tuple(IA...), A, tuple(IA...), conjA, α)
end
# implement for iterables
function tensorcopy(IC, A, IA, conjA::Symbol=:N, α::Number=true)
    return tensorcopy(tuple(IC...), A, tuple(IA...), conjA, α)
end
# expert mode
function tensorcopy(pC::Index2Tuple, A, conjA::Symbol=:N, α::Number=true,
                    backend::Backend...)
    TC = promote_add(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, pC, A, conjA)
    return tensorcopy!(C, pC, A, conjA, α, backend...)
end

"""
    tensorcopy!(C, pC::Index2Tuple, A, conjA=:N, α=true, [backend])

Copy the contents of tensor `A` into `C`, where the dimensions `A` are permuted according to
the permutation and repartition `pC`.

The result of this method is equivalent to `α * permutedims!(C, A, pC)`.

Optionally, the symbol `conjA` can be used to specify whether the input tensor should be
conjugated (`:C`) or not (`:N`).

!!! warning 
    The object `C` must not be aliased with `A`.

See also [`tensorcopy`](@ref) and [`tensoradd!`](@ref)
"""
function tensorcopy!(C, pC::Index2Tuple, A, conjA::Symbol=:N, α::Number=true,
                     backend::Backend...)
    return tensoradd!(C, pC, A, conjA, α, false, backend...)
end

# ------------------------------------------------------------------------------------------
# tensoradd
# ------------------------------------------------------------------------------------------

"""
    tensoradd([IC=IA], A, IA, [conjA], B, IB, [conjB], [α=true, [β=true]])
    tensoradd(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, pB::Index2Tuple, conjB, α=true, β=true, [backend]) # expert mode

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
                   conjB::Symbol, α::Number=true, β::Number=true)
    return tensoradd(A, add_indices(IA, IC), conjA, B, add_indices(IB, IC), conjB, α, β)
end
# default `IC`
function tensoradd(A, IA, conjA::Symbol, B, IB, conjB::Symbol,
                   α::Number=true, β::Number=true)
    return tensoradd(tuple(IA...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α, β)
end
# default `conjA` and `conjB`
function tensoradd(IC, A, IA, B, IB, α::Number=true, β::Number=true)
    return tensoradd(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α, β)
end
# default `IC`, `conjA` and `conjB`
function tensoradd(A, IA, B, IB, α::Number=true, β::Number=true)
    return tensoradd(tuple(IA...), A, tuple(IA...), B, tuple(IB...), α, β)
end
# iterables
function tensoradd(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol,
                   α::Number=true, β::Number=true)
    return tensoradd(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α, β)
end
# expert mode
function tensoradd(A, pA::Index2Tuple, conjA::Symbol,
                   B, pB::Index2Tuple, conjB::Symbol,
                   α::Number=true, β::Number=true, backend::Backend...)
    TC = promote_add(scalartype(A), scalartype(B), scalartype(α), scalartype(β))
    C = tensoralloc_add(TC, pA, A, conjA)
    C = tensorcopy!(C, pA, A, conjA, α)
    return tensoradd!(C, pB, B, conjB, β, true, backend...)
end

# ------------------------------------------------------------------------------------------
# tensortrace
# ------------------------------------------------------------------------------------------

"""
    tensortrace([IC], A, IA, [conjA], [α=true])
    tensortrace(pC::Index2Tuple, A, pA::Index2Tuple, conjA, α=true, [backend]) # expert mode

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

function tensortrace(IC::Tuple, A, IA::Tuple, conjA::Symbol=:N, α::Number=true)
    pC, cindA1, cindA2 = trace_indices(IA, IC)
    return tensortrace(pC, A, (cindA1, cindA2), conjA, α)
end
# default `IC`
function tensortrace(A, IA, conjA::Symbol, α::Number=true)
    return tensortrace(unique2(tuple(IA...)), A, tuple(IA...), conjA, α)
end
# default `conjA`
function tensortrace(IC, A, IA, α::Number=true)
    return tensortrace(tuple(IC...), A, tuple(IA...), :N, α)
end
# default `IC` and `conjA`
function tensortrace(A, IA, α::Number=true)
    return tensortrace(unique2(tuple(IA...)), A, tuple(IA...), :N, α)
end
# iterables
function tensortrace(IC, A, IA, conjA::Symbol, α::Number=true)
    return tensortrace(tuple(IC...), A, tuple(IA...), conjA, α)
end
# expert mode
function tensortrace(pC::Index2Tuple, A, pA::Index2Tuple, conjA::Symbol, α::Number=true,
                     backend::Backend...)
    TC = promote_contract(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, pC, A, conjA)
    return tensortrace!(C, pC, A, pA, conjA, α, false)
end

# ------------------------------------------------------------------------------------------
# tensorcontract
# ------------------------------------------------------------------------------------------

"""
    tensorcontract([IC], A, IA, [conjA], B, IB, [conjB], [α=true])
    tensorcontract(pC::Index2Tuple, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, α=true, [backend]) # expert mode

Contract indices of tensor `A` with corresponding indices in tensor `B` by assigning
them identical labels in the iterables `IA` and `IB`. The indices of the resulting
tensor correspond to the indices that only appear in either `IA` or `IB` and can be
ordered by specifying the optional argument `IC`. The default is to have all open
indices of `A` followed by all open indices of `B`. Note that inner contractions of an array
should be handled first with `tensortrace`, so that every label can appear only once in `IA`
or `IB` seperately, and once (for an open index) or twice (for a contracted index) in the
union of `IA` and `IB`.

The contraction can be performed by a native Julia algorithm without creating any
temporaries, or by first permuting the tensors such that the contraction becomes equivalent
to a matrix product, which is then performed by BLAS. The latter is typically faster for
large arrays with `BlasFloat` elements, while the former offers more flexibility when these
conditions are not met. The choice of method is globally controlled by the methods
[`enable_blas()`](@ref) and [`disable_blas()`](@ref), but the native algorithm is always
used when BLAS is not available.

Optionally, the symbols `conjA` and `conjB` can be used to specify that the input tensors
should be conjugated.

See also [`tensorcontract!`](@ref).
"""
function tensorcontract end

function tensorcontract(IC::Tuple, A, IA::Tuple, conjA::Symbol, B, IB::Tuple, conjB::Symbol,
                        α::Number=true)
    pA, pB, pC = contract_indices(IA, IB, IC)
    return tensorcontract(pC, A, pA, conjA, B, pB, conjB, α)
end
# default `IC`
function tensorcontract(A, IA, conjA, B, IB, conjB, α::Number=true)
    return tensorcontract(symdiff(tuple(IA...), tuple(IB...)), A, tuple(IA...), conjA, B,
                          tuple(IB...), conjB, α)
end
# default `conjA` and `conjB`
function tensorcontract(IC, A, IA, B, IB, α::Number=true)
    return tensorcontract(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α)
end
# default `IC`, `conjA` and `conjB`
function tensorcontract(A, IA, B, IB, α::Number=true)
    return tensorcontract(symdiff(tuple(IA...), tuple(IB...)), A, tuple(IA...), :N, B,
                          tuple(IB...), :N, α)
end
# iterables
function tensorcontract(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=true)
    return tensorcontract(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α)
end
# expert mode
function tensorcontract(pC::Index2Tuple, A, pA::Index2Tuple, conjA::Symbol, B,
                        pB::Index2Tuple, conjB::Symbol, α::Number=true, backend::Backend...)
    TC = promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = tensoralloc_contract(TC, pC, A, pA, conjA, B, pB, conjB)
    return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, false, backend...)
end

# ------------------------------------------------------------------------------------------
# tensorproduct
# ------------------------------------------------------------------------------------------

"""
    tensorproduct([IC], A, IA, [conjA], B, IB, [conjB], [α=true])
    tensorproduct(pC::Index2Tuple, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, α=true, [backend]) # expert mode

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
                       α::Number=true)
    pA, pB, pC = contract_indices(IA, IB, IC)
    return tensorproduct(pC, A, pA, conjA, B, pB, conjB, α)
end
# default `IC`
function tensorproduct(A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=true)
    return tensorproduct(vcat(tuple(IA...), tuple(IB...)), A, tuple(IA...), conjA, B,
                         tuple(IB...), conjB, α)
end
# default `conjA` and `conjB`
function tensorproduct(IC, A, IA, B, IB, α::Number=true)
    return tensorproduct(tuple(IC...), A, tuple(IA...), :N, B, tuple(IB...), :N, α)
end
# default `IC`, `conjA` and `conjB`
function tensorproduct(A, IA, B, IB, α::Number=true)
    return tensorproduct(vcat(tuple(IA...), tuple(IB...)), A, tuple(IA...), :N, B,
                         tuple(IB...), :N, α)
end
# iterables
function tensorproduct(IC, A, IA, conjA::Symbol, B, IB, conjB::Symbol, α::Number=true)
    return tensorproduct(tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB, α)
end
# expert mode
function tensorproduct(pC::Index2Tuple, A, pA::Index2Tuple, conjA::Symbol, B,
                       pB::Index2Tuple, conjB::Symbol, α::Number=true, backend::Backend...)
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract(pC, A, pA, conjA, B, pB, conjB, α, backend...)
end

"""
    tensorproduct!(C, pC::Index2Tuple, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, α=true, β=false)

Compute the tensor product (outer product) of two tensors `A` and `B`, i.e. a wrapper of
[`tensorcontract!`](@ref) with no indices being contracted over. This method checks whether
the indices indeed specify a tensor product instead of a genuine contraction.

!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See als [`tensorproduct`](@ref) and [`tensorcontract!`](@ref).
"""
function tensorproduct!(C, pC::Index2Tuple,
                        A, pA::Index2Tuple, conjA::Symbol,
                        B, pB::Index2Tuple, conjB::Symbol,
                        α::Number=true, β::Number=false, backend::Backend...)
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, backend...)
end
