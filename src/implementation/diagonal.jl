#-------------------------------------------------------------------------------------------
# Specialized implementations for contractions involving diagonal matrices
#-------------------------------------------------------------------------------------------
function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::Diagonal, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         ::StridedNative, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    if conjA && conjB
        _diagtensorcontract!(wrap_stridedview(C), conj(wrap_stridedview(A)), pA,
                             conj(wrap_stridedview(B.diag)), pB,
                             pAB, α, β)
    elseif conjA
        _diagtensorcontract!(wrap_stridedview(C), conj(wrap_stridedview(A)), pA,
                             wrap_stridedview(B.diag),
                             pB, pAB, α,
                             β)
    elseif conjB
        _diagtensorcontract!(wrap_stridedview(C), wrap_stridedview(A), pA,
                             conj(wrap_stridedview(B.diag)),
                             pB, pAB, α,
                             β)
    else
        _diagtensorcontract!(wrap_stridedview(C), wrap_stridedview(A), pA,
                             wrap_stridedview(B.diag), pB, pAB, α, β)
    end
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::Diagonal, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         ::StridedNative, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    rpA = reverse(pA)
    rpB = reverse(pB)
    indCinoBA = let N₁ = numout(pA), N₂ = numin(pB)
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), linearize(pAB))
    end
    tpAB = trivialpermutation(pAB)
    rpAB = (TupleTools.getindices(indCinoBA, tpAB[1]),
            TupleTools.getindices(indCinoBA, tpAB[2]))

    if conjA && conjB
        _diagtensorcontract!(wrap_stridedview(C), conj(wrap_stridedview(B)), rpB,
                             conj(wrap_stridedview(A.diag)), rpA, rpAB, α, β)
    elseif conjA
        _diagtensorcontract!(wrap_stridedview(C), wrap_stridedview(B), rpB,
                             conj(wrap_stridedview(A.diag)), rpA, rpAB, α, β)
    elseif conjB
        _diagtensorcontract!(wrap_stridedview(C), conj(wrap_stridedview(B)), rpB,
                             wrap_stridedview(A.diag), rpA, rpAB, α, β)
    else
        _diagtensorcontract!(wrap_stridedview(C), wrap_stridedview(B), rpB,
                             wrap_stridedview(A.diag), rpA, rpAB, α, β)
    end
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::Diagonal, pA::Index2Tuple, conjA::Bool,
                         B::Diagonal, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         ::StridedNative, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    if conjA && conjB
        _diagdiagcontract!(wrap_stridedview(C), conj(wrap_stridedview(A.diag)), pA,
                           conj(wrap_stridedview(B.diag)), pB, pAB, α, β)
    elseif conjA
        _diagdiagcontract!(wrap_stridedview(C), conj(wrap_stridedview(A.diag)), pA,
                           wrap_stridedview(B.diag), pB, pAB, α, β)
    elseif conjB
        _diagdiagcontract!(wrap_stridedview(C), wrap_stridedview(A.diag), pA,
                           conj(wrap_stridedview(B.diag)), pB, pAB, α, β)
    else
        _diagdiagcontract!(wrap_stridedview(C), wrap_stridedview(A.diag), pA,
                           wrap_stridedview(B.diag), pB, pAB, α, β)
    end
    return C
end

function tensorcontract!(C::Diagonal,
                         A::Diagonal, pA::Index2Tuple, conjA::Bool,
                         B::Diagonal, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number,
                         ::StridedNative, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    A2 = StridedView(A.diag)
    B2 = StridedView(B.diag)
    C2 = StridedView(C.diag)

    if conjA && conjB
        C2 .= C2 .* β .+ conj.(A2 .* B2) .* α
    elseif conjA
        C2 .= C2 .* β .+ conj.(A2) .* B2 .* α
    elseif conjB
        C2 .= C2 .* β .+ A2 .* conj.(B2) .* α
    else
        C2 .= C2 .* β .+ A2 .* B2 .* α
    end
    return C
end

function _diagtensorcontract!(C::StridedView,
                              A::StridedView, pA::Index2Tuple,
                              Bdiag::StridedView, pB::Index2Tuple,
                              pAB::Index2Tuple, α::Number, β::Number)
    sizeA = i -> size(A, i)
    csizeA = sizeA.(pA[2])
    osizeA = sizeA.(pA[1])

    if numin(pB) == 1 # => numin(A) == numout(B) == 1
        totsize = (osizeA..., csizeA...)
        A2 = permutedims(A, linearize(pA))
        B2 = sreshape(Bdiag, ((one.(osizeA))..., csizeA...))
        C2 = permutedims(C, invperm(linearize(pAB)))

    elseif numin(pB) == 0
        strideA = i -> stride(A, i)
        newstrides = (strideA.(pA[1])..., strideA(pA[2][1]) + strideA(pA[2][2]))
        totsize = (osizeA..., csizeA[1])
        A2 = StridedView(A.parent, totsize, newstrides, A.offset, A.op)
        B2 = sreshape(Bdiag, ((one.(osizeA))..., csizeA[1]))
        C2 = permutedims(C, invperm(linearize(pAB)))

    else # numout(pB) == 2 # direct product
        scale!(C, β)
        β = one(β)
        A2 = sreshape(permutedims(A, linearize(pA)), (osizeA..., 1))
        B2 = sreshape(Bdiag, ((one.(osizeA))..., length(Bdiag)))

        C3 = permutedims(C, invperm(linearize(pAB)))
        sC = strides(C3)
        newstrides = (Base.front(Base.front(sC))..., sC[end - 1] + sC[end])
        totsize = (osizeA..., length(Bdiag))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    op1 = Base.Fix2(scale, α) ∘ *
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, totsize, (C2, A2, B2))

    return C
end

function _diagdiagcontract!(C::StridedView,
                            Adiag::StridedView, pA::Index2Tuple,
                            Bdiag::StridedView, pB::Index2Tuple,
                            pAB::Index2Tuple, α::Number, β::Number)
    if numin(pA) == 1 # matrix multiplication
        scale!(C, β)
        β = one(β)

        A2 = sreshape(Adiag, (length(Adiag), 1))
        B2 = sreshape(Bdiag, (length(Bdiag), 1))
        # take a view of the diagonal elements of C, having strides 1 + length(diag)
        totsize = (length(Adiag),)
        C2 = StridedView(C.parent, totsize, (sum(strides(C)),))

    elseif numin(pA) == 2 # trace
        A2 = Adiag
        B2 = Bdiag
        totsize = (length(Adiag),)
        C2 = sreshape(C, (1,))

    else # outer product
        scale!(C, β)
        β = one(β)

        A2 = sreshape(Adiag, (length(Adiag), 1))
        B2 = sreshape(Bdiag, (1, length(Adiag)))

        C3 = permutedims(C, invperm(linearize(pAB)))
        strC = strides(C3)
        newstrides = (strC[1] + strC[2], strC[3] + strC[4])
        totsize = (length(A2), length(B2))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    op1 = Base.Fix2(scale, α) ∘ *
    op2 = Base.Fix2(scale, β)
    Strided._mapreducedim!(op1, +, op2, totsize, (C2, A2, B2))

    return C
end
