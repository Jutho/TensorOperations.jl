const StridedViewOrDiagonal = Union{StridedView,Diagonal}

_ishostarray(x::StridedView) = (pointer(x) isa Ptr)
_ishostarray(x::Diagonal) = (pointer(x.diag) isa Ptr)

function select_backend(::typeof(tensoradd!), C::StridedView, A::StridedView)
    if _ishostarray(C) && _ishostarray(A)
        return StridedNative()
    else
        return NoBackend()
    end
end
function select_backend(::typeof(tensortrace!), C::StridedView, A::StridedView)
    if _ishostarray(C) && _ishostarray(A)
        return StridedNative()
    else
        return NoBackend()
    end
end

function select_backend(::typeof(tensorcontract!), C::StridedView, A::StridedView,
                        B::StridedView)
    if _ishostarray(C) && _ishostarray(A) && _ishostarray(B)
        return eltype(C) <: LinearAlgebra.BlasFloat ? StridedBLAS() : StridedNative()
    else
        return NoBackend()
    end
end
function select_backend(::typeof(tensorcontract!), C::StridedViewOrDiagonal,
                        A::StridedViewOrDiagonal, B::StridedViewOrDiagonal)
    if _ishostarray(C) && _ishostarray(A) && _ishostarray(B)
        return StridedNative()
    else
        return NoBackend()
    end
end

#-------------------------------------------------------------------------------------------
# Force strided implementation on AbstractArray instances with Strided backend
#-------------------------------------------------------------------------------------------
const SV = StridedView
function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number,
                    backend::StridedBackend, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensoradd!(SV(C), conj(SV(A)), pA, α, β, backend, allocator)
    else
        stridedtensoradd!(SV(C), SV(A), pA, α, β, backend, allocator)
    end
    return C
end

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number,
                      backend::StridedBackend, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensortrace!(SV(C), conj(SV(A)), p, q, α, β, backend, allocator)
    else
        stridedtensortrace!(SV(C), SV(A), p, q, α, β, backend, allocator)
    end
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         backend::StridedBackend, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA && conjB
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    elseif conjA
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    elseif conjB
        stridedtensorcontract!(SV(C), SV(A), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    else
        stridedtensorcontract!(SV(C), SV(A), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    end
    return C
end

#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
function stridedtensoradd!(C::StridedView,
                           A::StridedView, pA::Index2Tuple,
                           α::Number, β::Number,
                           ::StridedBackend, allocator=DefaultAllocator())
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    if !istrivialpermutation(pA) && Base.mightalias(C, A)
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    end

    A′ = permutedims(A, linearize(pA))
    op1 = Base.Fix2(scale, α)
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, size(C), (C, A′))
    return C
end

function stridedtensortrace!(C::StridedView,
                             A::StridedView, p::Index2Tuple, q::Index2Tuple,
                             α::Number, β::Number,
                             ::StridedBackend, allocator=DefaultAllocator())
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    sizeA = i -> size(A, i)
    strideA = i -> stride(A, i)
    tracesize = sizeA.(q[1])
    newstrides = (strideA.(linearize(p))..., (strideA.(q[1]) .+ strideA.(q[2]))...)
    newsize = (size(C)..., tracesize...)

    A′ = SV(A.parent, newsize, newstrides, A.offset, A.op)
    op1 = Base.Fix2(scale, α)
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, newsize, (C, A′))
    return C
end

function stridedtensorcontract!(C::StridedView,
                                A::StridedView, pA::Index2Tuple,
                                B::StridedView, pB::Index2Tuple,
                                pAB::Index2Tuple,
                                α::Number, β::Number,
                                ::StridedBLAS, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    rpA = reverse(pA)
    rpB = reverse(pB)
    indCinoBA = let N₁ = numout(pA), N₂ = numin(pB)
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), linearize(pAB))
    end
    tpAB = trivialpermutation(pAB)
    rpAB = (TupleTools.getindices(indCinoBA, tpAB[1]),
            TupleTools.getindices(indCinoBA, tpAB[2]))
    if contract_memcost(C, A, pA, B, pB, pAB) <= contract_memcost(C, B, rpB, A, rpA, rpAB)
        return blas_contract!(C, A, pA, B, pB, pAB, α, β, allocator)
    else
        return blas_contract!(C, B, rpB, A, rpA, rpAB, α, β, allocator)
    end
    return C
end

# reduce overhead for the case where it is just matrix multiplication
function stridedtensorcontract!(C::StridedView{T,2},
                                A::StridedView{T,2}, pA::Index2Tuple{1,1},
                                B::StridedView{T,2}, pB::Index2Tuple{1,1},
                                pAB::Index2Tuple{1,1}, α::Number, β::Number,
                                ::StridedBLAS) where {T}
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    A′ = pA == ((1,), (2,)) ? A : permutedims(A, (pA[1][1], pA[2][1]))
    B′ = pB == ((1,), (2,)) ? B : permutedims(B, (pB[1][1], pB[2][1]))
    if pAB == ((1,), (2,))
        mul!(C, A′, B′, α, β)
    elseif pAB == ((2,), (1,))
        mul!(C, transpose(B′), transpose(A′), α, β)
    end
    return C
end

function stridedtensorcontract!(C::StridedView,
                                A::StridedView, pA::Index2Tuple,
                                B::StridedView, pB::Index2Tuple,
                                pAB::Index2Tuple,
                                α::Number, β::Number,
                                ::StridedNative, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    AS = sreshape(permutedims(A, linearize(pA)), (osizeA..., one.(osizeB)..., csizeA...))
    BS = sreshape(permutedims(B, linearize(reverse(pB))),
                  (one.(osizeA)..., osizeB..., csizeB...))
    CS = sreshape(permutedims(C, invperm(linearize(pAB))),
                  (osizeA..., osizeB..., one.(csizeA)...))
    tsize = (osizeA..., osizeB..., csizeA...)

    op1 = Base.Fix2(scale, α) ∘ *
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, tsize, (CS, AS, BS))
    return C
end

#-------------------------------------------------------------------------------------------
# StridedViewBLAS contraction implementation
#-------------------------------------------------------------------------------------------
function blas_contract!(C, A, pA, B, pB, pAB, α, β, allocator)
    TC = eltype(C)

    A_, pA, flagA = makeblascontractable(A, pA, TC, allocator)
    B_, pB, flagB = makeblascontractable(B, pB, TC, allocator)

    ipAB = oindABinC(pAB, pA, pB)
    flagC = isblasdestination(C, ipAB)
    if flagC
        C_ = C
        _unsafe_blas_contract!(C_, A_, pA, B_, pB, ipAB, α, β)
    else
        C_ = SV(tensoralloc_add(TC, C, ipAB, false, Val(true), allocator))
        _unsafe_blas_contract!(C_, A_, pA, B_, pB, trivialpermutation(ipAB),
                               one(TC), zero(TC))
        stridedtensoradd!(C, C_, pAB, α, β, StridedNative(), allocator)
        tensorfree!(C_.parent, allocator)
    end
    flagA || tensorfree!(A_.parent, allocator)
    flagB || tensorfree!(B_.parent, allocator)
    return C
end

function _unsafe_blas_contract!(C::StridedView{T},
                                A::StridedView{T}, pA,
                                B::StridedView{T}, pB,
                                pAB, α, β) where {T<:BlasFloat}
    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    mul!(sreshape(permutedims(C, linearize(pAB)), (prod(osizeA), prod(osizeB))),
         sreshape(permutedims(A, linearize(pA)), (prod(osizeA), prod(csizeA))),
         sreshape(permutedims(B, linearize(pB)), (prod(csizeB), prod(osizeB))),
         α, β)

    return C
end

@inline function makeblascontractable(A, pA, TC, allocator)
    flagA = isblascontractable(A, pA) && eltype(A) == TC
    if !flagA
        A_ = tensoralloc_add(TC, A, pA, false, Val(true), allocator)
        Anew = SV(A_, size(A_), strides(A_), 0, A.op)
        Anew = stridedtensoradd!(Anew, A, pA, One(), Zero(), StridedNative(), allocator)
        pAnew = trivialpermutation(pA)
    else
        Anew = A
        pAnew = pA
    end
    return Anew, pAnew, flagA
end

function isblascontractable(A::StridedView, p::Index2Tuple)
    eltype(A) <: LinearAlgebra.BlasFloat || return false

    sizeA = size(A)
    stridesA = strides(A)
    sizeA1 = TupleTools.getindices(sizeA, p[1])
    sizeA2 = TupleTools.getindices(sizeA, p[2])
    stridesA1 = TupleTools.getindices(stridesA, p[1])
    stridesA2 = TupleTools.getindices(stridesA, p[2])

    canfuse1, _, s1 = _canfuse(sizeA1, stridesA1)
    canfuse2, _, s2 = _canfuse(sizeA2, stridesA2)

    if A.op == conj
        return canfuse1 && canfuse2 && s2 == 1
    else
        return canfuse1 && canfuse2 && (s1 == 1 || s2 == 1)
    end
end

function isblasdestination(A::StridedView, p::Index2Tuple)
    (eltype(A) <: LinearAlgebra.BlasFloat && A.op == identity) || return false

    sizeA = size(A)
    stridesA = strides(A)

    sizeA1 = TupleTools.getindices(sizeA, p[1])
    stridesA1 = TupleTools.getindices(stridesA, p[1])
    canfuse1, _, s1 = _canfuse(sizeA1, stridesA1)
    (canfuse1 && s1 == 1) || return false

    sizeA2 = TupleTools.getindices(sizeA, p[2])
    stridesA2 = TupleTools.getindices(stridesA, p[2])
    canfuse2, _, _ = _canfuse(sizeA2, stridesA2)
    return canfuse2
end

_canfuse(::Dims{0}, ::Dims{0}) = true, 1, 1
_canfuse(dims::Dims{1}, strides::Dims{1}) = true, dims[1], strides[1]
@inline function _canfuse(dims::Dims{N}, strides::Dims{N}) where {N}
    if dims[1] == 0
        return true, 0, 1
    elseif dims[1] == 1
        return _canfuse(Base.tail(dims), Base.tail(strides))
    else
        b, d, s = _canfuse(Base.tail(dims), Base.tail(strides))
        if b && (s == dims[1] * strides[1] || d == 1)
            dnew = dims[1] * d
            return true, dnew, (dnew == 0 || dnew == 1) ? 1 : strides[1]
        else
            return false, dims[1] * d, strides[1]
        end
    end
end

function oindABinC(pAB, pA, pB)
    ipAB = invperm(linearize(pAB))
    oindAinC = TupleTools.getindices(ipAB, trivialpermutation(pA[1]))
    oindBinC = TupleTools.getindices(ipAB, numout(pA) .+ trivialpermutation(pB[2]))
    return (oindAinC, oindBinC)
end

function contract_memcost(C, A, pA, B, pB, pAB)
    ipAB = oindABinC(pAB, pA, pB)
    return length(A) * (!isblascontractable(A, pA) || eltype(A) !== eltype(C)) +
           length(B) * (!isblascontractable(B, pB) || eltype(B) !== eltype(C)) +
           length(C) * !isblasdestination(C, ipAB)
end
