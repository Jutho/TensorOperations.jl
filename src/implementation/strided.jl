#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
function tensoradd!(C::StridedView,
                    A::StridedView, pA::Index2Tuple, conjA::Symbol,
                    α::Number, β::Number,
                    backend::Union{StridedNative,StridedBLAS}=StridedNative())
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    if !istrivialpermutation(pA) && Base.mightalias(C, A)
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    end

    A′ = permutedims(flag2op(conjA)(A), linearize(pA))
    op1 = Base.Fix2(scale, α)
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, size(C), (C, A′))
    return C
end

function tensortrace!(C::StridedView,
                      A::StridedView, p::Index2Tuple, q::Index2Tuple, conjA::Symbol,
                      α::Number, β::Number,
                      backend::Union{StridedNative,StridedBLAS}=StridedNative())
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    sizeA = i -> size(A, i)
    strideA = i -> stride(A, i)
    tracesize = sizeA.(q[1])
    newstrides = (strideA.(linearize(p))..., (strideA.(q[1]) .+ strideA.(q[2]))...)
    newsize = (size(C)..., tracesize...)

    A′ = flag2op(conjA)(StridedView(A.parent, newsize, newstrides, A.offset, A.op))
    op1 = Base.Fix2(scale, α)
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, newsize, (C, A′))
    return C
end

function tensorcontract!(C::StridedView{T},
                         A::StridedView, pA::Index2Tuple, conjA::Symbol,
                         B::StridedView, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         backend::StridedBLAS=StridedBLAS()) where {T<:LinearAlgebra.BlasFloat}
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
    if contract_memcost(C, A, pA, conjA, B, pB, conjB, pAB) <=
       contract_memcost(C, B, rpB, conjB, A, rpA, conjA, rpAB)
        return blas_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
    else
        return blas_contract!(C, B, rpB, conjB, A, rpA, conjA, rpAB, α, β)
    end
end

# reduce overhead for the case where it is just matrix multiplication
function tensorcontract!(C::StridedView{T,2},
                         A::StridedView{T,2}, pA::Index2Tuple{1,1}, conjA::Symbol,
                         B::StridedView{T,2}, pB::Index2Tuple{1,1}, conjB::Symbol,
                         pAB::Index2Tuple{1,1}, α::Number, β::Number,
                         backend::StridedBLAS=StridedBLAS()) where {T<:LinearAlgebra.BlasFloat}
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    opA = flag2op(conjA)
    opB = flag2op(conjB)
    A′ = pA == ((1,), (2,)) ? opA(A) : opA(permutedims(A, (pA[1][1], pA[2][1])))
    B′ = pB == ((1,), (2,)) ? opB(B) : opB(permutedims(B, (pB[1][1], pB[2][1])))
    if pAB == ((1,), (2,))
        mul!(C, A′, B′, α, β)
    elseif pAB == ((2,), (1,))
        mul!(C, transpose(B′), transpose(A′), α, β)
    end
    return C
end

function tensorcontract!(C::StridedView,
                         A::StridedView, pA::Index2Tuple, conjA::Symbol,
                         B::StridedView, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple, α::Number, β::Number,
                         backend::StridedNative)
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    AS = sreshape(permutedims(flag2op(conjA)(A), linearize(pA)),
                  (osizeA..., one.(osizeB)..., csizeA...))
    BS = sreshape(permutedims(flag2op(conjB)(B), linearize(reverse(pB))),
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
function blas_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)
    TC = eltype(C)

    A_, pA, conjA, flagA = makeblascontractable(A, pA, conjA, TC)
    B_, pB, conjB, flagB = makeblascontractable(B, pB, conjB, TC)

    ipAB = oindABinC(pAB, pA, pB)
    flagC = isblascontractable(C, ipAB, :D)
    if flagC
        C_ = C
        _unsafe_blas_contract!(C_, A_, pA, conjA, B_, pB, conjB, ipAB, α, β)
    else
        C_ = StridedView(TensorOperations.tensoralloc_add(TC, C, ipAB, :N, true))
        _unsafe_blas_contract!(C_, A_, pA, conjA, B_, pB, conjB, trivialpermutation(ipAB),
                               one(TC), zero(TC))
        tensoradd!(C, C_, pAB, :N, α, β)
        tensorfree!(C_.parent)
    end
    flagA || tensorfree!(A_.parent)
    flagB || tensorfree!(B_.parent)
    return C
end

function _unsafe_blas_contract!(C::StridedView{T},
                                A::StridedView{T}, pA, conjA,
                                B::StridedView{T}, pB, conjB,
                                pAB, α, β) where {T<:BlasFloat}
    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    opA = flag2op(conjA)
    opB = flag2op(conjB)

    mul!(sreshape(permutedims(C, linearize(pAB)), (prod(osizeA), prod(osizeB))),
         opA(sreshape(permutedims(A, linearize(pA)), (prod(osizeA), prod(csizeA)))),
         opB(sreshape(permutedims(B, linearize(pB)), (prod(csizeB), prod(osizeB)))),
         α, β)

    return C
end

@inline function makeblascontractable(A, pA, conjA, TC)
    flagA = isblascontractable(A, pA, conjA) && eltype(A) == TC
    if !flagA
        A_ = StridedView(TensorOperations.tensoralloc_add(TC, A, pA, conjA, true))
        A = tensoradd!(A_, A, pA, conjA, One(), Zero())
        conjA = :N
        pA = trivialpermutation(pA)
    end
    return A, pA, conjA, flagA
end

function isblascontractable(A::StridedView, p::Index2Tuple, C::Symbol)
    eltype(A) <: LinearAlgebra.BlasFloat || return false
    sizeA = size(A)
    stridesA = strides(A)
    sizeA1 = TupleTools.getindices(sizeA, p[1])
    sizeA2 = TupleTools.getindices(sizeA, p[2])
    stridesA1 = TupleTools.getindices(stridesA, p[1])
    stridesA2 = TupleTools.getindices(stridesA, p[2])

    canfuse1, d1, s1 = _canfuse(sizeA1, stridesA1)
    canfuse2, d2, s2 = _canfuse(sizeA2, stridesA2)

    if C == :D # destination
        return A.op == identity && canfuse1 && canfuse2 && s1 == 1
    elseif (C == :C && A.op == identity) || (C == :N && A.op == conj) # conjugated
        return canfuse1 && canfuse2 && s2 == 1
    else
        return canfuse1 && canfuse2 && (s1 == 1 || s2 == 1)
    end
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

function contract_memcost(C, A, pA, conjA, B, pB, conjB, pAB)
    ipAB = oindABinC(pAB, pA, pB)
    return length(A) *
           (!isblascontractable(A, pA, conjA) || eltype(A) !== eltype(C)) +
           length(B) *
           (!isblascontractable(B, pB, conjB) || eltype(B) !== eltype(C)) +
           length(C) * !isblascontractable(C, ipAB, :D)
end
