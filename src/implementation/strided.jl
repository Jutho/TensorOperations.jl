tensorscalar(C::AbstractArray) = ndims(C) == 0 ? C[] : throw(DimensionMismatch())

function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                    α, β)
    ndims(C) == ndims(A) || throw(DimensionMismatch("ndims(A) ≠ ndims(C)"))
    ndims(C) == length(pA[1]) + length(pA[2]) ||
        throw(IndexError("Invalid permutation of length $(ndims(C)): $pA"))
    
    C === A && throw(ArgumentError("output tensor must not be aliased with input tensor"))

    if conjA == :N
        add!(StridedView(C), permutedims(StridedView(A), linearize(pA)), α, β)
    elseif conjA == :C
        add!(StridedView(C), permutedims(conj(StridedView(A)), linearize(pA)), α, β)
    elseif conjA == :A
        add!(StridedView(C), permutedims(adjoint(StridedView(A)), linearize(pA)), α, β)
    else
        throw(ArgumentError("unknown conjugation flag: $conjA"))
    end
    return C
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         α, β)
    (length(pA[1]) + length(pA[2]) == ndims(A) && TupleTools.isperm(linearize(pA))) ||
        throw(IndexError("invalid permutation of A of length $(ndims(A)): $pA"))
    (length(pB[1]) + length(pB[2]) == ndims(B) && TupleTools.isperm(linearize(pB))) ||
        throw(IndexError("invalid permutation of B of length $(ndims(B)): $pB"))
    (length(pA[1]) + length(pB[2]) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (length(pC[1]) + length(pC[2]) == ndims(C) && TupleTools.isperm(linearize(pC))) ||
        throw(IndexError("invalid permutation of C of length $(ndims(C)): $pC"))
    
    (C === A || C === B) && throw(ArgumentError("output tensor must not be aliased with input tensor"))
        
    szA = size(A)
    szB = size(B)
    szC = size(C)

    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))

    szoA = TupleTools.getindices(szA, pA[1])
    szoB = TupleTools.getindices(szB, pB[2])
    szoC = TupleTools.getindices(szC, linearize(pC))
    szoC = TupleTools.getindices(szC, TupleTools.invperm(linearize(pC)))
    TupleTools.getindices((szoA..., szoB...), linearize(pC)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions:" *
                                "($szoA, $szoB)[$(linearize(pC))] -> $szC"))

    if use_blas() && eltype(C) <: LinearAlgebra.BlasFloat &&
       !isa(B, Diagonal) && !isa(A, Diagonal)
        if contract_memcost(C, pC, A, pA, conjA, B, pB, conjB) >
           reversecontract_memcost(C, pC, A, pA, conjA, B, pB, conjB)
            return blas_reversecontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
        else
            return blas_contract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
        end
    else
        return native_contract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
    end
end

function tensortrace!(C::AbstractArray, pC::Index2Tuple,
                      A::AbstractArray, pA::Index2Tuple, conjA::Symbol, α,
                      β)
    NA, NC = ndims(A), ndims(C)
    NC == length(linearize(pC)) ||
        throw(IndexError("invalid selection of $NC out of $NA: $pC"))
    NA - NC == 2 * length(pA[1]) == 2 * length(pA[2]) ||
        throw(IndexError("invalid number of trace dimensions"))

    C === A && throw(ArgumentError("output tensor must not be aliased with input tensor"))

    inds = ((linearize(pC)...,), pA[1], pA[2])
    if conjA == :N
        _trace!(α, StridedView(A), β, StridedView(C), inds...)
    elseif conjA == :C
        _trace!(α, conj(StridedView(A)), β, StridedView(C), inds...)
    elseif conjA == :A
        _trace!(α, map(adjoint, StridedView(A)), β, StridedView(C), inds...)
    else
        throw(ArgumentError("Unknown conjugation flag: $conjA"))
    end

    return C
end

function blas_contract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
    TC = eltype(C)

    A_, pA, conjA, flagA = makeblascontractable(A, pA, conjA, TC)
    B_, pB, conjB, flagB = makeblascontractable(B, pB, conjB, TC)

    pC_ = oindABinC(pC, pA, pB)

    flagC = isblascontractable(C, pC_[1], pC_[2], :D)
    if flagC
        C_ = C
        _blas_contract!(α, A_, conjA, B_, conjB, β, C_, pA, pB, pC_)
    else
        C_ = TensorOperations.tensoralloc_add(TC, C, pC_, :N)
        pC__ = (_trivtuple(pA[1]), length(pA[1]) .+ _trivtuple(pB[2]))
        _blas_contract!(1, A_, conjA, B_, conjB, 0, C_, pA, pB, pC__)
        tensoradd!(C, C_, pC, :N, α, β)
    end

    flagA || tensorfree!(A_)
    flagB || tensorfree!(B_)
    flagC || tensorfree!(C_)

    return C
end

function blas_reversecontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
    indCinoBA = let N₁ = length(pA[1]), N₂ = length(pB[2])
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), linearize(pC))
    end
    pC_BA = (TupleTools.getindices(indCinoBA, _trivtuple(pC[1])),
             TupleTools.getindices(indCinoBA, length(pC[1]) .+ _trivtuple(pC[2])))
    return blas_contract!(C, pC_BA, B, reverse(pB), conjB, A, reverse(pA), conjA, α, β)
end

function makeblascontractable(A, pA, conjA, TC)
    flagA = isblascontractable(A, pA[1], pA[2], conjA) && eltype(A) == TC
    if !flagA
        A_ = TensorOperations.tensoralloc_add(TC, A, pA, conjA)
        A = tensoradd!(A_, A, pA, conjA, one(TC), zero(TC))
        conjA = :N
        pA = (_trivtuple(pA[1]), _trivtuple(pA[2]) .+ length(pA[1]))
    end
    return A, pA, conjA, flagA
end

function _blas_contract!(α, A::AbstractArray, conjA, B::AbstractArray, conjB, β,
                         C::AbstractArray, pA, pB, pC)
    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    pABinC = oindABinC(pC, pA, pB)
    mul!(sreshape(permutedims(StridedView(C), linearize(pABinC)),
                  (prod(osizeA), prod(osizeB))),
         flag2op(conjA)(sreshape(permutedims(StridedView(A), linearize(pA)),
                                 (prod(osizeA), prod(csizeA)))),
         flag2op(conjB)(sreshape(permutedims(StridedView(B), linearize(pB)),
                                 (prod(csizeB), prod(osizeB)))),
         α, β)

    return C
end

function native_contract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
    sizeA = size(A)
    sizeB = size(B)
    csizeA = TupleTools.getindices(sizeA, pA[2])
    csizeB = TupleTools.getindices(sizeB, pB[1])
    osizeA = TupleTools.getindices(sizeA, pA[1])
    osizeB = TupleTools.getindices(sizeB, pB[2])

    let opA = flag2op(conjA), opB = flag2op(conjB), α = α
        AS = sreshape(permutedims(StridedView(A), linearize(pA)),
                      (osizeA..., one.(osizeB)..., csizeA...))
        BS = sreshape(permutedims(StridedView(B), linearize(reverse(pB))),
                      (one.(osizeA)..., osizeB..., csizeB...))
        CS = sreshape(permutedims(StridedView(C), TupleTools.invperm(linearize(pC))),
                      (osizeA..., osizeB..., one.(csizeA)...))
        tsize = (osizeA..., osizeB..., csizeA...)

        if α != 1
            op1 = (x, y) -> α * opA(x) * opB(y)
            if β == 0
                Strided._mapreducedim!(op1, +, zero, tsize, (CS, AS, BS))
            elseif β == 1
                Strided._mapreducedim!(op1, +, nothing, tsize, (CS, AS, BS))
            else
                Strided._mapreducedim!(op1, +, y -> β * y, tsize, (CS, AS, BS))
            end
        else
            op2 = (x, y) -> opA(x) * opB(y)
            if β == 0
                if isbitstype(eltype(C))
                    Strided._mapreducedim!(op2, +, zero, tsize, (CS, AS, BS))
                else
                    fill!(C, zero(eltype(C)))
                    Strided._mapreducedim!(op2, +, nothing, tsize, (CS, AS, BS))
                end
            elseif β == 1
                Strided._mapreducedim!(op2, +, nothing, tsize, (CS, AS, BS))
            else
                Strided._mapreducedim!(op2, +, y -> β * y, tsize, (CS, AS, BS))
            end
        end
    end

    return C
end

function native_contract!(C, pC, A::AbstractArray, pA, conjA, B::Diagonal, pB, conjB, α, β)
    Bd = B.diag
    oindA, cindA = pA
    cindB, oindB = pB
    indCinoAB = linearize(pC)

    @strided begin
        if conjA == :N && conjB == :N
            _contract!(α, A, Bd, β, C, oindA, cindA, oindB, cindB, indCinoAB)
        elseif conjA == :C && conjB == :N
            _contract!(α, conj(A), Bd, β, C, oindA, cindA, oindB, cindB, indCinoAB)
        elseif conjA == :N && conjB == :C
            _contract!(α, A, conj(Bd), β, C, oindA, cindA, oindB, cindB, indCinoAB)
        elseif conjA == :C && conjB == :C
            _contract!(α, conj(A), conj(Bd), β, C, oindA, cindA, oindB, cindB, indCinoAB)
        else
            throw(ArgumentError("unknown conjugation flag $conjA and $conjB"))
        end
    end

    return C
end

function native_contract!(C, pC, A::Diagonal, pA, conjA, B::AbstractArray, pB, conjB, α, β)
    indCinoBA = let N₁ = length(pA[1]), N₂ = length(pB[2])
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), linearize(pC))
    end
    pC_BA = (TupleTools.getindices(indCinoBA, _trivtuple(pC[1])),
             TupleTools.getindices(indCinoBA, length(pC[1]) .+ _trivtuple(pC[2])))
    return native_contract!(C, pC_BA, B, reverse(pB), conjB, A, reverse(pA), conjA, α, β)
end

function _contract!(α, A::StridedView, Bd::StridedView,
                    β, C::StridedView, oindA, cindA, oindB, cindB, indCinoAB)
    sizeA = i -> size(A, i)
    csizeA = sizeA.(cindA)
    osizeA = sizeA.(oindA)

    if length(oindB) == 1 # length(cindA) == length(cindB) == 1
        A2 = permutedims(A, (oindA..., cindA...))
        C2 = permutedims(C, TupleTools.invperm(indCinoAB))
        B2 = sreshape(Bd, ((one.(osizeA))..., csizeA...))
        totsize = (osizeA..., csizeA...)

    elseif length(oindB) == 0
        strideA = i -> stride(A, i)
        newstrides = (strideA.(oindA)..., strideA(cindA[1]) + strideA(cindA[2]))
        totsize = (osizeA..., csizeA[1])
        A2 = StridedView(A.parent, totsize, newstrides, A.offset, A.op)
        B2 = sreshape(Bd, ((one.(osizeA))..., csizeA[1]))
        C2 = permutedims(C, TupleTools.invperm(indCinoAB))

    else # length(oindB) == 2
        if β != 1
            rmul!(C, β)
            β = 1
        end
        A2 = sreshape(permutedims(A, (oindA..., cindA...)), (osizeA..., 1))
        C3 = permutedims(C, TupleTools.invperm(indCinoAB))
        B2 = sreshape(Bd, ((one.(osizeA))..., length(Bd)))

        sC = strides(C3)
        newstrides = (Base.front(Base.front(sC))..., sC[end - 1] + sC[end])
        totsize = (osizeA..., length(Bd))

        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    op1 = α == 1 ? (*) : (x, y) -> α * x * y
    op2 = β == 0 ? zero : β == 1 ? nothing : y -> β * y
    Strided._mapreducedim!(op1, +, op2, totsize, (C2, A2, B2))

    return C
end

function flag2op(flag::Symbol)
    op = flag == :N ? identity :
         flag == :C ? conj :
         flag == :A ? adjoint :
         throw(ArgumentError("unknown conjuagation flag $flag"))
    return op
end

function _trace!(α, A::StridedView,
                 β, C::StridedView, indCinA::IndexTuple{NC},
                 cindA1::IndexTuple{NT}, cindA2::IndexTuple{NT}) where {NC,NT}
    sizeA = i -> size(A, i)
    strideA = i -> stride(A, i)
    tracesize = sizeA.(cindA1)
    tracesize == sizeA.(cindA2) || throw(DimensionMismatch("non-matching trace sizes"))
    size(C) == sizeA.(indCinA) || throw(DimensionMismatch("non-matching sizes"))

    newstrides = (strideA.(indCinA)..., (strideA.(cindA1) .+ strideA.(cindA2))...)
    newsize = (size(C)..., tracesize...)
    A2 = StridedView(A.parent, newsize, newstrides, A.offset, A.op)

    if α != 1
        if β == 0
            Strided._mapreducedim!(x -> α * x, +, zero, newsize, (C, A2))
        elseif β == 1
            Strided._mapreducedim!(x -> α * x, +, nothing, newsize, (C, A2))
        else
            Strided._mapreducedim!(x -> α * x, +, y -> β * y, newsize, (C, A2))
        end
    else
        if β == 0
            return Strided._mapreducedim!(identity, +, zero, newsize, (C, A2))
        elseif β == 1
            Strided._mapreducedim!(identity, +, nothing, newsize, (C, A2))
        else
            Strided._mapreducedim!(identity, +, y -> β * y, newsize, (C, A2))
        end
    end
    return C
end

function isblascontractable(A::AbstractArray, p1::IndexTuple, p2::IndexTuple,
                            C::Symbol)
    eltype(A) <: LinearAlgebra.BlasFloat || return false
    @strided isblascontractable(A, p1, p2, C)
end

function isblascontractable(A::StridedView, p1::IndexTuple, p2::IndexTuple,
                            C::Symbol)
    eltype(A) <: LinearAlgebra.BlasFloat || return false
    sizeA = size(A)
    stridesA = strides(A)
    sizeA1 = TupleTools.getindices(sizeA, p1)
    sizeA2 = TupleTools.getindices(sizeA, p2)
    stridesA1 = TupleTools.getindices(stridesA, p1)
    stridesA2 = TupleTools.getindices(stridesA, p2)

    canfuse1, d1, s1 = _canfuse(sizeA1, stridesA1)
    canfuse2, d2, s2 = _canfuse(sizeA2, stridesA2)

    if C == :D # destination
        return A.op == identity && canfuse1 && canfuse2 && s1 == 1
    elseif (C == :C && A.op == identity) || (C == :N && A.op == conj)# conjugated
        return canfuse1 && canfuse2 && s2 == 1
    else
        return canfuse1 && canfuse2 && (s1 == 1 || s2 == 1)
    end
end

_canfuse(::Dims{0}, ::Dims{0}) = true, 1, 1
_canfuse(dims::Dims{1}, strides::Dims{1}) = true, dims[1], strides[1]
function _canfuse(dims::Dims{N}, strides::Dims{N}) where {N}
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
_trivtuple(::NTuple{N}) where {N} = ntuple(identity, Val(N))

function oindABinC(pC, pA, pB)
    ipC = TupleTools.invperm(linearize(pC))
    oindAinC = TupleTools.getindices(ipC, _trivtuple(pA[1]))
    oindBinC = TupleTools.getindices(ipC, length(pA[1]) .+ _trivtuple(pB[2]))
    return oindAinC, oindBinC
end

function contract_memcost(C, pC, A, pA, conjA, B, pB, conjB)
    ipC = oindABinC(pC, pA, pB)
    return length(A) *
           (!isblascontractable(A, pA[1], pA[2], conjA) || eltype(A) !== eltype(C)) +
           length(B) *
           (!isblascontractable(B, pB[1], pB[2], conjB) || eltype(B) !== eltype(C)) +
           length(C) * !isblascontractable(C, ipC[1], ipC[2], :D)
end

function reversecontract_memcost(C, pC, A, pA, conjA, B, pB, conjB)
    return contract_memcost(C, reverse(pC), B, reverse(pB), conjB, A, reverse(pA), conjA)
end