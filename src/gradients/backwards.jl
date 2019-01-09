# gradients/backwards.jl
#
# Gradient functions called in backward pass, can be re-used for any framework.
# Note that I haven't thought about complex numbers at all, so conjA etc may be wrong.

const ∇VERBOSE = false # debugging

function add∇A(Δ, α, A::TA, conjA, β, C::TC, indCinA) where {TA,TC}
    add!(α, Δ, conjA, 0, similar(data(A)), invperm(indCinA)) 
end

function any∇C(Δ, β)
    β .* Δ
end

function trace∇A(Δ, α, A::TA, conjA, β, C::TC, indCinA, cindA1, cindA2) where {TA,TC}

    # csize = ntuple(i -> size(A,cindA1[i]), length(cindA1))
    # T = eltype(Δ) # note that this is called with data(Δ)
    # K = dirac(T, (csize..., csize...)) # default type Bool here much slower
    K = dirac!(cached_similar_from_indices(:dirac, eltype(Δ), cindA1, cindA2, A, :N))

    ∇VERBOSE && @info "...trace∇A..." size(A) (cindA1,cindA2) indCinA size(K) # csize

    indK = ntuple(i->i, 2*length(cindA1))
    indΔ = ntuple(i->i, ndims(Δ))
    indAinoKΔ = TupleTools.invperm((cindA1..., cindA2..., indCinA...))

    # simA = similar(A)
    indA = ntuple(i->i, ndims(A))
    simA = similar_from_indices(eltype(A), indA, (), A, :N)

    ∇A = contract!(α, K, :N, Δ, conjA, false, simA, indK, (), indΔ, (), indAinoKΔ)
end

# Trying to use similar_from_indices ... for dirac I can use cache,
# and for contract∇A I add something to the given symbols, should be unique,
# does it matter that a matrix from the cache may be returned as A.grad?

# Could do likewise in ∇add() below, and ∇C = β .* Δ
# It would be neat if trace! and add! were also given a syms argument by the @tensor macro.

findint(n::Int, tup::Tuple)::Int = findfirst(i->i==n, tup)

function contract∇A(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms=nothing)

    indAinoΔB_old = ntuple(i->i, ndims(A))
    indAinoΔB = TupleTools.invperm((oindA..., cindA...))
    ∇VERBOSE && println("indAinoΔB_old = ",indAinoΔB_old, "  , indAinoΔB = ",indAinoΔB)
    oindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))
    cindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))

    ∇VERBOSE && @info "...∇A..." indAinoΔB (oindΔ, cindΔ)  (cindB, oindB) syms

    # simA = similar(A)
    indA = ntuple(i->i, ndims(A))
    simA = cached_similar_from_indices(sym_glue(syms, :_c∇A), eltype(A), indA, (), A, :N)

    ∇A = contract!(α, Δ, conjA, B, conjB, false, simA, oindΔ, cindΔ, cindB, oindB, indAinoΔB, sym_suffix(syms, :_∇A))
end

function contract∇B(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms=nothing)

    indBinoAΔ = TupleTools.invperm((cindB..., oindB...))
    oindΔ = ntuple(i -> findint(i+length(oindA), indCinoAB), length(oindB))
    cindΔ = ntuple(i -> findint(i, indCinoAB), length(oindA))

    ∇VERBOSE && @info "...∇B..." indBinoAΔ (oindΔ, cindΔ)  (cindA, oindA) syms

    # simB = similar(B)
    indB = ntuple(i->i, ndims(B))
    simB = cached_similar_from_indices(sym_glue(syms, :_c∇B), eltype(B), indB, (), B, :N)

    ∇B = contract!(α, A, conjA, Δ, conjB, false, simB, cindA, oindA, oindΔ, cindΔ, indBinoAΔ, sym_suffix(syms, :_∇B))
end

sym_suffix(syms, suffix) = Symbol.(syms, suffix)
sym_suffix(::Nothing, suffix) = nothing

sym_glue(syms, suffix) = Symbol(syms..., suffix)
sym_glue(::Nothing, suffix) = Symbol(:Δnew, suffix)


add∇α(Δ, α, A, conjA, β, C, indCinA) = (@warn "add∇α not yet defined"; false) # dot(Δ, permutedims(A...)) yuck
add∇β(Δ, α, A, conjA, β, C, indCinA) = (@warn "add∇β not yet defined"; false) # dot(Δ, C_orig) but that's been overwritten

trace∇α(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) = (@warn "trace∇α not yet defined"; false)
trace∇β(Δ, α, A, conjA, β, C, indCinA, cindA1, cindA2) = (@warn "trace∇β not yet defined"; false)

contract∇α(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms) = (@warn "contract∇α not yet defined"; false)
contract∇β(Δ, α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms) = (@warn "contract∇β not yet defined"; false)

using LinearAlgebra

"""
    dirac([T,] size)
Dense array of the given size, describing the product of `n = length(size)/2` kronecker deltas,
which equate the first `n` indices with the last `n`.
For `n=1` this is simply `Matrix{T}(I, size)`, with `T=Bool` by default.
For `n=2` it is `D[i,j,k,l] = i==k && j==l`, and so on.

    dirac!(A)
Given an array, this fills it with `0` and `1` as above.
"""
dirac(size::Tuple) = dirac(Bool, size)
dirac(x::T, size::Tuple) where {T<:Number} = dirac(T, size)

dirac(T::Type, size::NTuple{2,Int}) = Matrix{T}(LinearAlgebra.I, size)
dirac(T::Type, size::Tuple) = dirac_fill!(zeros(T, size), pairstep(cumprod1(size)), pairmins(size))

@doc @doc(dirac)
function dirac!(a::AbstractArray{T,N}) where {T,N}
    @assert iseven(N) "dirac! needs an even number of array indices"
    a .= zero(T)
    dirac_fill!(a, pairstep(cumprod1(size(a))), pairmins(size(a)))
end

cumprod1(tup::NTuple{N,T}) where {N,T} = ntuple(i -> i==1 ? one(T) : prod(tup[j] for j=1:i-1), Val(N))
pairstep(tup::NTuple{N,T}) where {N,T} = ntuple(i -> tup[i] + tup[i+N÷2], Val(N÷2))
pairmins(tup::NTuple{N,T}) where {N,T} = ntuple(i -> min(tup[i], tup[i+N÷2]), Val(N÷2))

using Base.Cartesian

@generated function dirac_fill!(array::AbstractArray{T,N}, steps::NTuple{D}, stops) where {T,N,D}
    quote
        @nloops $D  i  k->1:stops[k]  begin
            lin = 1
            @nexprs  $D  k->(@inbounds lin += steps[k] * (i_k - 1))
            @inbounds array[lin] = one($T)
        end
        array
    end
end
