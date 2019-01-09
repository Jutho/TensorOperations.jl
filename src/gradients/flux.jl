# gradients/flux.jl
#
# Connect up gradients for Flux's TrackedArrays. 

using .Flux
using .Flux.Tracker: track, @grad, TrackedArray, TrackedReal, data

using Strided
import Strided: StridedView, UnsafeStridedView

# similar_from_indices always makes an un-tracked array, as tracking is handled outside its concerns

similar_from_indices(::Type{Flux.Tracker.TrackedReal{T}}, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol) where T =
    similar_from_indices(T, p1, p2, data(A), CA)

cached_similar_from_indices(sym::Symbol, ::Type{Flux.Tracker.TrackedReal{T}}, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol) where T =
    cached_similar_from_indices(sym, T, p1, p2, data(A), CA)

similar_from_indices(::Type{Flux.Tracker.TrackedReal{T}}, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol) where T =
    similar_from_indices(T, poA, poB, p1, p2, data(A), data(B), CA, CB)

cached_similar_from_indices(sym::Symbol, ::Type{Flux.Tracker.TrackedReal{T}}, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple, A, B, CA::Symbol, CB::Symbol) where T =
    cached_similar_from_indices(sym, T, poA, poB, p1, p2, data(A), data(B), CA, CB)

StridedView(A::Flux.Tracker.TrackedArray) = StridedView(A.data)
UnsafeStridedView(A::Flux.Tracker.TrackedArray) = UnsafeStridedView(A.data)

function promote_type_α(T, Tα::TrackedReal{Tr}) where {Tr}
    ∇VERBOSE && @info "promote_type_α" T Tr
    promote_type(T, Tr)
end


# Track the these basic functions

add!(α, A::TrackedArray{TA,N}, conjA::Symbol, β, C::AbstractArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA)
add!(α, A::Array{TA,N}, conjA::Symbol, β, C::TrackedArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA) # case of A untracked
add!(α, A::TrackedArray{TA,N}, conjA::Symbol, β, C::TrackedArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA) # because of method ambiguity

add!(α::TrackedReal, A::AbstractArray{TA,N}, conjA::Symbol, β, C::AbstractArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA) # arises from promotion... which ideally would be delayed a bit?
add!(α::TrackedReal, A::TrackedArray{TA,N}, conjA::Symbol, β, C::AbstractArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA)
add!(α::TrackedReal, A::Array{TA,N}, conjA::Symbol, β, C::TrackedArray{TC,N}, indCinA) where {TA,TC,N} =
    track(add!, α, A, conjA, β, C, indCinA)

# In v0.7, you only got α::TrackedReal when this was explicitly supplied, and I made it an error in ∇add.
# Now it occurs due to promotion too. As a result I must track more cases, to avoid Float64(TrackedReal) errors.

trace!(α, A::TrackedArray, conjA::Symbol, β, C::AbstractArray, indCinA, cindA1, cindA2) =
    track(trace!, α, A, conjA, β, C, indCinA, cindA1, cindA2)

trace!(α::TrackedReal, A::TrackedArray, conjA::Symbol, β, C::AbstractArray, indCinA, cindA1, cindA2) =
    track(trace!, α, A, conjA, β, C, indCinA, cindA1, cindA2)


contract!(α, A::TrackedArray, conjA::Symbol, B::AbstractArray, conjB::Symbol, β, C::AbstractArray,
    oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple,
    indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
    track(contract!, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
contract!(α, A::Array, conjA::Symbol, B::TrackedArray, conjB::Symbol, β, C::AbstractArray,
    oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple,
    indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
    track(contract!, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)

contract!(α::TrackedReal, A::TrackedArray, conjA::Symbol, B::AbstractArray, conjB::Symbol, β, C::AbstractArray,
    oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple,
    indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
    track(contract!, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
contract!(α::TrackedReal, A::Array, conjA::Symbol, B::TrackedArray, conjB::Symbol, β, C::AbstractArray,
    oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple,
    indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
    track(contract!, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)


# Corresponding _forward definitions

@grad function add!(α, A, conjA, β, C, indCinA)
    ∇VERBOSE && @info "@grad add!" α summary(A) A[1] conjA β summary(C) C[1] indCinA
    add!(data(α), data(A), conjA, data(β), data(C), indCinA),
        Δ -> ∇add(Δ, α, A, conjA, β, C, indCinA) # not data() yet, so that ∇add knows which to compute
end

#       track(trace!, α, A, conjA, β, C, indCinA, cindA1, cindA2)
@grad function trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)
    ∇VERBOSE && @info "@grad trace!" α summary(A) A[1] conjA β summary(C) C[1] indCinA cindA1 cindA2
    trace!(data(α), data(A), conjA, data(β), data(C), indCinA, cindA1, cindA2),
        Δ -> ∇trace(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) # not data() yet
end

@grad function contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    ∇VERBOSE && @info "@grad contract! #1" α summary(A) A[1] conjA summary(B) B[1] conjB β summary(C) C[1] oindA cindA oindB cindB indCinoAB syms
    contract!(data(α), data(A), conjA, data(B), conjB, data(β), data(C), oindA, cindA, oindB, cindB, indCinoAB, syms),
        Δ -> ∇contract(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms) # not data() yet
end

# Backward pass functions

function ∇add(Δ, α::Tα, A::TA, conjA, β::Tβ, C::TC, indCinA) where {Tα,TA,Tβ,TC}
    ∇VERBOSE && @info "∇add" summary(Δ) Δ[1] α summary(A) A[1] conjA β summary(C) C[1] indCinA

    ∇A = TA<:TrackedArray ? add∇A(data(Δ), data(α), A, conjA, β, C, indCinA) : nothing
    ∇C = TC<:TrackedArray ? data(β) .* data(Δ) : nothing

    ∇α = false # Tα<:TrackedReal ? add∇α(Δ, α, A, conjA, β, C, indCinA) : false
    ∇β = false # Tβ<:TrackedReal ? add∇β(Δ, α, A, conjA, β, C, indCinA) : false

    return (∇α, ∇A, nothing, ∇β, ∇C, nothing)
end

function ∇trace(Δ, α::Tα, A::TA, conjA, β::Tβ, C::TC, indCinA, cindA1, cindA2) where {Tα,TA,Tβ,TC}
    ∇VERBOSE && @info "∇trace" summary(Δ) Δ[1] α summary(A) A[1] conjA β summary(C) C[1] indCinA cindA1 cindA2

    ∇A = TA<:TrackedArray ? trace∇A(data(Δ), data(α), data(A), conjA, data(β), data(C), indCinA, cindA1, cindA2) : nothing
    ∇C = TC<:TrackedArray ? data(β) .* data(Δ) : nothing

    ∇α = false # Tα<:TrackedReal ? trace∇α(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) : false
    ∇β = false # Tβ<:TrackedReal ? trace∇β(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) : false

    return (∇α, ∇A, nothing, ∇β, ∇C, nothing, nothing, nothing)
end

function ∇contract(Δ, α::Tα, A::TA, conjA, B::TB, conjB, β::Tβ, C::TC, oindA, cindA, oindB, cindB, indCinoAB, syms) where {Tα,TA,Tβ,TB,TC}
    ∇VERBOSE && @info "∇contract" summary(Δ) Δ[1] α summary(A) A[1] conjA summary(B) B[1] conjB β summary(C) C[1] oindA cindA oindB cindB indCinoAB syms

    ∇A = TA<:TrackedArray ?
        contract∇A(Δ, data(α), data(A), conjA, data(B), conjB, data(β), data(C), oindA, cindA, oindB, cindB, indCinoAB, syms) : nothing
    ∇B = TB<:TrackedArray ?
        contract∇B(Δ, data(α), data(A), conjA, data(B), conjB, data(β), data(C), oindA, cindA, oindB, cindB, indCinoAB, syms) : nothing
    ∇C = TC<:TrackedArray ? data(β) .* data(Δ) : nothing

    ∇α = false # Tα<:TrackedReal ?
        # contract∇α(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB) : false
    ∇β = false # Tβ<:TrackedReal ?
        # contract∇β(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB) : false

    return (∇α, ∇A, nothing, ∇B, nothing, ∇β, ∇C, nothing, nothing, nothing, nothing, nothing, nothing)
end

# Note that I haven't allowed for α,β to be tracked.
# Besides writing these functions, it would some more _forward definitions,
# and perhaps copying of the input matrices, and lots more tests!
#
# In v0.7 TensorOperations, these contract∇α etc. were never called if you didn't explicitly pass α::TrackedReal,
# so I made errors to warn you.
# But in v1 TensorOperations, α gets promoted sometimes to eltype(A) and thus these would be called more often,
# even when not required, so for now the are simply never called.
# This change also made dispatch of add!() etc more complicated, see above.

