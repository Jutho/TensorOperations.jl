#-------------------------------------------------------------------------------------------
# Specialized implementations for contractions involving diagonal matrices
#-------------------------------------------------------------------------------------------
function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::Diagonal, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedNative)
    argcheck_tensorcontract(C, pC, A, pA, B, pB)
    dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    _diagtensorcontract!(StridedView(C), pC, StridedView(A), pA, conjA,
                         StridedView(B.diag), pB, conjB, α, β)
    return C
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::Diagonal, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedNative)
    argcheck_tensorcontract(C, pC, A, pA, B, pB)
    dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    rpA = reverse(pA)
    rpB = reverse(pB)
    indCinoBA = let N₁ = numout(pA), N₂ = numin(pB)
        map(n -> ifelse(n > N₁, n - N₁, n + N₂), linearize(pC))
    end
    tpC = trivialpermutation(pC)
    rpC = (TupleTools.getindices(indCinoBA, tpC[1]),
           TupleTools.getindices(indCinoBA, tpC[2]))

    _diagtensorcontract!(StridedView(C), rpC, StridedView(B), rpB, conjB,
                         StridedView(A.diag), rpA, conjA, α, β)
    return C
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::Diagonal, pA::Index2Tuple, conjA::Symbol,
                         B::Diagonal, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedNative)
    argcheck_tensorcontract(C, pC, A, pA, B, pB)
    dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    if numin(pA) == 1 # matrix multiplication
        if β != one(β) # multiplication only takes care of diagonal elements of C
            rmul!(C, β)
            β = one(β)
        end

        A2 = sreshape(flag2op(conjA)(StridedView(A.diag)), (length(A.diag), 1))
        B2 = sreshape(flag2op(conjB)(StridedView(B.diag)), (length(B.diag), 1))
        # take a view of the diagonal elements of C, having strides 1 + length(diag)
        totsize = (length(A.diag),)
        C2 = StridedView(C, totsize, (sum(strides(C)),))

    elseif numin(pA) == 2 # trace
        A2 = flag2op(conjA)(StridedView(A.diag, (length(A.diag),)))
        B2 = flag2op(conjB)(StridedView(B.diag, (length(B.diag),)))
        totsize = (length(A.diag),)
        C2 = sreshape(StridedView(C), (1,))

    else # outer product
        if β != 1
            rmul!(C, β)
            β = 1
        end

        A2 = sreshape(StridedView(A.diag), (length(A.diag), 1))
        B2 = sreshape(StridedView(B.diag), (1, length(A.diag)))

        C3 = permutedims(StridedView(C), invperm(linearize(pC)))
        strC = strides(C3)
        newstrides = (strC[1] + strC[2], strC[3] + strC[4])
        totsize = (length(A2), length(B2))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    op1 = α == 1 ? (*) : (x, y) -> α * x * y
    op2 = β == 0 ? zero : β == 1 ? nothing : y -> β * y
    Strided._mapreducedim!(op1, +, op2, totsize, (C2, A2, B2))

    return C
end

function tensorcontract!(C::Diagonal, pC::Index2Tuple,
                         A::Diagonal, pA::Index2Tuple, conjA::Symbol,
                         B::Diagonal, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedNative)
    argcheck_tensorcontract(C, pC, A, pA, B, pB)
    dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    A2 = flag2op(conjA)(StridedView(A.diag))
    B2 = flag2op(conjB)(StridedView(B.diag))
    C2 = StridedView(C.diag)

    C2 .= α .* A2 .* B2 .+ β .* C2
    return C
end

function _diagtensorcontract!(C::StridedView, pC::Index2Tuple,
                              A::StridedView, pA::Index2Tuple, conjA::Symbol,
                              Bdiag::StridedView, pB::Index2Tuple, conjB::Symbol,
                              α, β)
    sizeA = i -> size(A, i)
    csizeA = sizeA.(pA[2])
    osizeA = sizeA.(pA[1])

    if numin(pB) == 1 # => numin(A) == numout(B) == 1
        totsize = (osizeA..., csizeA...)
        A2 = flag2op(conjA)(permutedims(A, linearize(pA)))
        B2 = flag2op(conjB)(sreshape(Bdiag, ((one.(osizeA))..., csizeA...)))
        C2 = permutedims(C, invperm(linearize(pC)))

    elseif numin(pB) == 0
        strideA = i -> stride(A, i)
        newstrides = (strideA.(pA[1])..., strideA(pA[2][1]) + strideA(pA[2][2]))
        totsize = (osizeA..., csizeA[1])
        A2 = flag2op(conjA)(StridedView(A.parent, totsize, newstrides, A.offset, A.op))
        B2 = flag2op(conjB)(sreshape(Bdiag, ((one.(osizeA))..., csizeA[1])))
        C2 = permutedims(C, invperm(linearize(pC)))

    else # numout(pB) == 2 # direct product
        if β != 1
            rmul!(C, β)
            β = 1
        end
        A2 = flag2op(conjA)(sreshape(permutedims(A, linearize(pA)), (osizeA..., 1)))
        B2 = flag2op(conjB)(sreshape(Bdiag, ((one.(osizeA))..., length(Bdiag))))

        C3 = permutedims(C, invperm(linearize(pC)))
        sC = strides(C3)
        newstrides = (Base.front(Base.front(sC))..., sC[end - 1] + sC[end])
        totsize = (osizeA..., length(Bdiag))
        C2 = StridedView(C3.parent, totsize, newstrides, C3.offset, C3.op)
    end

    op1 = α == 1 ? (*) : (x, y) -> α * x * y
    op2 = β == 0 ? zero : β == 1 ? nothing : y -> β * y
    Strided._mapreducedim!(op1, +, op2, totsize, (C2, A2, B2))

    return C
end
