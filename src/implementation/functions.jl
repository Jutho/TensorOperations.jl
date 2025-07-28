# methods/simple.jl
#
# Method-based access to tensor operations using simple definitions.

# utility function
function _kwargs2args(; kwargs...)
    for k in keys(kwargs)
        if !(k in (:backend, :allocator))
            throw(ArgumentError("unknown keyword argument: $k"))
        end
    end
    if haskey(kwargs, :backend) && haskey(kwargs, :allocator)
        return (kwargs[:backend], kwargs[:allocator])
    elseif haskey(kwargs, :allocator)
        return (DefaultBackend(), kwargs[:allocator])
    elseif haskey(kwargs, :backend)
        return (kwargs[:backend],)
    else
        return ()
    end
end

# ------------------------------------------------------------------------------------------
# tensorcopy
# ------------------------------------------------------------------------------------------
"""
    tensorcopy([IC=IA], A, IA, [conjA=false, [α=1]]; [backend=..., allocator=...])
    tensorcopy(A, pA::Index2Tuple, conjA, α, [backend, allocator]) # expert mode

Create a copy of `A`, where the dimensions of `A` are assigned indices from the
iterable `IA` and the indices of the copy are contained in `IC`. Both iterables
should contain the same elements, optionally in a different order.

The result of this method is equivalent to `α * permutedims(A, pA)` where `pA` is the
permutation such that `IC = IA[pA]`. The implementation of `tensorcopy` is however more
efficient on average, especially if `Threads.nthreads() > 1`.

The optional argument `conjA` can be used to specify whether the input tensor should be
conjugated (`true`) or not (`false`), whereas `α` can be used to scale the result. It is
also optional to specify a backend implementation to use, and an allocator to be used if
temporary tensor objects are needed.

See also [`tensorcopy!`](@ref).
"""
function tensorcopy end

function tensorcopy(
        IC::Labels, A, IA::Labels, conjA::Bool = false, α::Number = One();
        kwargs...
    )
    pA = add_indices(IA, IC)
    return tensorcopy(A, pA, conjA, α, _kwargs2args(; kwargs...)...)
end
# default `IC`
function tensorcopy(A, IA::Labels, conjA::Bool = false, α::Number = One(); kwargs...)
    return tensorcopy(IA, A, IA, conjA, α; kwargs...)
end
# expert mode
function tensorcopy(
        A, pA::Index2Tuple, conjA::Bool = false, α::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    TC = promote_add(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, A, pA, conjA, Val(false), allocator)
    return tensorcopy!(C, A, pA, conjA, α, backend, allocator)
end

"""
    tensorcopy!(C, A, pA::Index2Tuple, conjA=false, α=1, [backend, allocator])

Copy the contents of tensor `A` into `C`, where the dimensions `A` are permuted according to
the permutation and repartition `pA`.

The result of this method is equivalent to `α * permutedims!(C, A, pA)`.

Optionally, the flag `conjA` can be used to specify whether the input tensor should be
conjugated (`true`) or not (`false`). 

!!! warning 
    The object `C` must not be aliased with `A`.

See also [`tensorcopy`](@ref) and [`tensoradd!`](@ref)
"""
function tensorcopy!(
        C, A, pA::Index2Tuple, conjA::Bool = false, α::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    return tensoradd!(C, A, pA, conjA, α, Zero(), backend, allocator)
end

# ------------------------------------------------------------------------------------------
# tensoradd
# ------------------------------------------------------------------------------------------
"""
    tensoradd([IC=IA], A, IA, [conjA], B, IB, [conjB], [α=1, [β=1]]; [backend=..., allocator=...])
    tensoradd(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, α=1, β=1, [backend, allocator]) # expert mode

Return the result of adding arrays `A` and `B` where the iterables `IA` and `IB`
denote how the array data should be permuted in order to be added. More specifically,
the result of this method is equivalent to
`α * permutedims(A, pA) + β * permutedims(B, pB)` where `pA` (`pB`) is the permutation such
that `IC = IA[pA]` (`IB[pB]`). The implementation of `tensoradd` is however more efficient
on average, as the temporary permuted arrays are not created.

Optionally, the symbols `conjA` and `conjB` can be used to specify whether the input tensors
should be conjugated (`true`) or not (`false`).

See also [`tensoradd!`](@ref).
"""
function tensoradd end

function tensoradd(
        IC::Labels, A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool,
        α::Number = One(), β::Number = One();
        kwargs...
    )
    pA = add_indices(IA, IC)
    pB = add_indices(IB, IC)
    return tensoradd(A, pA, conjA, B, pB, conjB, α, β, _kwargs2args(; kwargs...)...)
end
# default `IC`
function tensoradd(
        A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool,
        α::Number = One(), β::Number = One();
        kwargs...
    )
    return tensoradd(IA, A, IA, conjA, B, IB, conjB, α, β; kwargs...)
end
# default `conjA` and `conjB`
function tensoradd(
        IC::Labels, A, IA::Labels, B, IB::Labels, α::Number = One(), β::Number = One();
        kwargs...
    )
    return tensoradd(IC, A, IA, false, B, IB, false, α, β)
end
# default `IC`, `conjA` and `conjB`
function tensoradd(
        A, IA::Labels, B, IB::Labels, α::Number = One(), β::Number = One();
        kwargs...
    )
    return tensoradd(IA, A, IA, B, IB, α, β; kwargs...)
end
# expert mode
function tensoradd(
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        α::Number = One(), β::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    TC = promote_add(scalartype(A), scalartype(B), scalartype(α), scalartype(β))
    C = tensoralloc_add(TC, A, pA, conjA, Val(false), allocator)
    C = tensorcopy!(C, A, pA, conjA, α, backend, allocator)
    return tensoradd!(C, B, pB, conjB, β, One(), backend, allocator)
end

# ------------------------------------------------------------------------------------------
# tensortrace
# ------------------------------------------------------------------------------------------
"""
    tensortrace([IC], A, IA, [conjA], [α=1]; [backend=..., allocator=...])
    tensortrace(A, p::Index2Tuple, q::Index2Tuple, conjA, α=1, [backend, allocator]) # expert mode

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

function tensortrace(IC::Labels, A, IA::Labels, conjA::Bool, α::Number = One(); kwargs...)
    p, q = trace_indices(IA, IC)
    return tensortrace(A, p, q, conjA, α, _kwargs2args(; kwargs...)...)
end
# default `IC`
function tensortrace(A, IA::Labels, conjA::Bool, α::Number = One(); kwargs...)
    return tensortrace(unique2(IA), A, IA, conjA, α; kwargs...)
end
# default `conjA`
function tensortrace(IC::Labels, A, IA::Labels, α::Number = One(); kwargs...)
    return tensortrace(IC, A, IA, false, α; kwargs...)
end
# default `IC` and `conjA`
function tensortrace(A, IA::Labels, α::Number = One(); kwargs...)
    return tensortrace(unique2(IA), A, IA, false, α; kwargs...)
end
# expert mode
function tensortrace(
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    TC = promote_contract(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, A, p, conjA, Val(false), allocator)
    return tensortrace!(C, A, p, q, conjA, α, Zero(), backend, allocator)
end

# ------------------------------------------------------------------------------------------
# tensorcontract
# ------------------------------------------------------------------------------------------
"""
    tensorcontract([IC], A, IA, [conjA], B, IB, [conjB], [α=1]; [backend=..., allocator=...])
    tensorcontract(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, [backend, allocator]) # expert mode

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

function tensorcontract(
        IC::Labels, A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool,
        α::Number = One();
        kwargs...
    )
    pA, pB, pAB = contract_indices(IA, IB, IC)
    return tensorcontract(A, pA, conjA, B, pB, conjB, pAB, α, _kwargs2args(; kwargs...)...)
end
# default `IC`
function tensorcontract(
        A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool,
        α::Number = One();
        kwargs...
    )
    return tensorcontract(symdiff(IA, IB), A, IA, conjA, B, IB, conjB, α; kwargs...)
end
# default `conjA` and `conjB`
function tensorcontract(
        IC::Labels, A, IA::Labels, B, IB::Labels, α::Number = One();
        kwargs...
    )
    return tensorcontract(IC, A, IA, false, B, IB, false, α; kwargs...)
end
# default `IC`, `conjA` and `conjB`
function tensorcontract(A, IA::Labels, B, IB::Labels, α::Number = One(); kwargs...)
    return tensorcontract(symdiff(IA, IB), A, IA, false, B, IB, false, α; kwargs...)
end
# expert mode
function tensorcontract(
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    TC = promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB, Val(false), allocator)
    return tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, Zero(), backend, allocator
    )
end

# ------------------------------------------------------------------------------------------
# tensorproduct
# ------------------------------------------------------------------------------------------

"""
    tensorproduct([IC], A, IA, [conjA], B, IB, [conjB], [α=1]; [backend=..., allocator=...])
    tensorproduct(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, [backend, allocator]) # expert mode

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

function tensorproduct(
        IC::Labels, A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool,
        α::Number = One();
        kwargs...
    )
    pA, pB, pAB = contract_indices(IA, IB, IC)
    return tensorproduct(A, pA, conjA, B, pB, conjB, pAB, α, _kwargs2args(; kwargs...)...)
end
# default `IC`
function tensorproduct(
        A, IA::Labels, conjA::Bool, B, IB::Labels, conjB::Bool, α::Number = One();
        kwargs...
    )
    return tensorproduct(vcat(IA, IB), A, IA, conjA, B, IB, conjB, α; kwargs...)
end
# default `conjA` and `conjB`
function tensorproduct(IC::Labels, A, IA::Labels, B, IB::Labels, α::Number = One(); kwargs...)
    return tensorproduct(IC, A, IA, false, B, IB, false, α; kwargs...)
end
# default `IC`, `conjA` and `conjB`
function tensorproduct(A, IA::Labels, B, IB::Labels, α::Number = One(); kwargs...)
    return tensorproduct(vcat(IA, IB), A, IA, false, B, IB, false, α; kwargs...)
end
# expert mode
function tensorproduct(
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number = One(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract(A, pA, conjA, B, pB, conjB, pAB, α, backend, allocator)
end

"""
    tensorproduct!(C, A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, β=0, [backend, allocator])

Compute the tensor product (outer product) of two tensors `A` and `B`, i.e. a wrapper of
[`tensorcontract!`](@ref) with no indices being contracted over. This method checks whether
the indices indeed specify a tensor product instead of a genuine contraction.  Optionally 
specify a backend implementation to use, and an allocator to be used if temporary tensor
objects are needed.


!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See als [`tensorproduct`](@ref) and [`tensorcontract!`](@ref).
"""
function tensorproduct!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number = One(), β::Number = Zero(),
        backend = DefaultBackend(), allocator = DefaultAllocator()
    )
    numin(pA) == 0 && numout(pB) == 0 ||
        throw(IndexError("not a valid tensor product"))
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator)
end
