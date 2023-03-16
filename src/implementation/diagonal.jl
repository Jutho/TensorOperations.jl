# overwrite similar from indices, because similar on `Diagonal` tends to create
# `SparseArray` for some horrible reason

# function similar_from_indices(T::Type, ind::IndexTuple, A::Diagonal, CA::Symbol)
#     sz = similarstructure_from_indices(T, ind, A, CA)
#     return similar(A.diag, T, sz)
# end
# function similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
#                               p1::IndexTuple, p2::IndexTuple,
#                               A::Diagonal, B::AbstractArray, CA::Symbol, CB::Symbol)
#     sz = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
#     return similar(A.diag, T, sz)
# end
# function similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
#                               p1::IndexTuple, p2::IndexTuple,
#                               A::Diagonal, B::Diagonal, CA::Symbol, CB::Symbol)
#     sz = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
#     return similar(A.diag, T, sz)
# end

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
