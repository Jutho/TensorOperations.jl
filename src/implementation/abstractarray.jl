tensorscalar(C::AbstractArray) = ndims(C) == 0 ? C[] : throw(DimensionMismatch())

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(A::AbstractArray, iA, conjA::Bool,
                           B::AbstractArray, iB, conjB::Bool, label)
    size(A, iA) == size(B, iB) ||
        throw(DimensionMismatch("Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
    return nothing
end

function select_backend(::typeof(tensoradd!), C::AbstractArray, A::AbstractArray)
    if isstrided(A) && isstrided(C)
        return StridedNative()
    else
        return BaseView()
    end
end

function select_backend(::typeof(tensortrace!), C::AbstractArray, A::AbstractArray)
    if isstrided(A) && isstrided(C)
        return StridedNative()
    else
        return BaseView()
    end
end

function select_backend(::typeof(tensorcontract!), C::AbstractArray, A::AbstractArray,
                        B::AbstractArray)
    if eltype(C) <: LinearAlgebra.BlasFloat
        if isstrided(A) && isstrided(B) && isstrided(C)
            return StridedBLAS()
        elseif (isstrided(A) || isa(A, Diagonal)) && (isstrided(B) || isa(B, Diagonal)) &&
               isstrided(C)
            return StridedNative()
        else
            return BaseCopy()
        end
    else
        if (isstrided(A) || isa(A, Diagonal)) && (isstrided(B) || isa(B, Diagonal)) &&
           isstrided(C)
            return StridedNative()
        else
            return BaseView()
        end
    end
end

#-------------------------------------------------------------------------------------------
# Implementation based on StridedViews
#-------------------------------------------------------------------------------------------

function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number, backend::StridedBackend)
    tensoradd!(StridedView(C), StridedView(A), pA, conjA, α, β, backend)
    return C
end

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number, backend::StridedBackend)
    tensortrace!(StridedView(C), StridedView(A), p, q, conjA, α, β, backend)
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::StridedBLAS)
    tensorcontract!(StridedView(C),
                    StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB,
                    pAB, α, β, StridedBLAS())
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::StridedNative)
    tensorcontract!(StridedView(C),
                    StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB,
                    pAB, α, β, StridedNative())
    return C
end

#-------------------------------------------------------------------------------------------
# Implementation based on Base + LinearAlgebra
#-------------------------------------------------------------------------------------------
# Note that this is mostly for convenience + checking, and not for performance
function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number, ::BaseView)
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    if conjA
        if iszero(β)
            C .= α .* conj.(PermutedDimsArray(A, linearize(pA)))
        else
            C .= β .* C .+ α .* conj.(PermutedDimsArray(A, linearize(pA)))
        end
    else
        if iszero(β)
            C .= α .* PermutedDimsArray(A, linearize(pA))
        else
            C .= β .* C .+ α .* PermutedDimsArray(A, linearize(pA))
        end
    end
    return C
end
function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number, ::BaseCopy)
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    if conjA
        if iszero(β)
            C .= α .* conj.(permutedims(A, linearize(pA)))
        else
            C .= β .* C .+ α .* conj.(permutedims(A, linearize(pA)))
        end
    else
        if iszero(β)
            C .= α .* permutedims(A, linearize(pA))
        else
            C .= β .* C .+ α .* permutedims(A, linearize(pA))
        end
    end
    return C
end

# tensortrace
function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number, ::BaseView)
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    szA = size(A)
    so = TupleTools.getindices(szA, linearize(p))
    st = prod(TupleTools.getindices(szA, q[1]))
    Ã = reshape(PermutedDimsArray(A, (linearize(p)..., linearize(q)...)),
                 (prod(so), st, st))

    if conjA
        if iszero(β)
            C .= α .* conj.(reshape(view(Ã, :, 1, 1), so))
        else
            C .= β .* C .+ α .* conj.(reshape(view(Ã, :, 1, 1), so))
        end
        for i in 2:st
            C .+= α .* conj.(reshape(view(Ã, :, i, i), so))
        end
    else
        if iszero(β)
            C .= α .* reshape(view(Ã, :, 1, 1), so)
        else
            C .= β .* C .+ α .* reshape(view(Ã, :, 1, 1), so)
        end
        for i in 2:st
            C .+= α .* reshape(view(Ã, :, i, i), so)
        end
    end
    return C
end
function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number, ::BaseCopy)
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    szA = size(A)
    so = TupleTools.getindices(szA, linearize(p))
    st = prod(TupleTools.getindices(szA, q[1]))
    Ã = reshape(permutedims(A, (linearize(p)..., linearize(q)...)), (prod(so), st, st))

    if conjA
        if iszero(β)
            C .= α .* conj.(reshape(view(Ã, :, 1, 1), so))
        else
            C .= β .* C .+ α .* conj.(reshape(view(Ã, :, 1, 1), so))
        end
        for i in 2:st
            C .+= α .* conj.(reshape(view(Ã, :, i, i), so))
        end
    else
        if iszero(β)
            C .= α .* reshape(view(Ã, :, 1, 1), so)
        else
            C .= β .* C .+ α .* reshape(view(Ã, :, 1, 1), so)
        end
        for i in 2:st
            C .+= α .* reshape(view(Ã, :, i, i), so)
        end
    end
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::BaseView)
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    szA = size(A)
    szB = size(B)
    soA = TupleTools.getindices(szA, pA[1])
    soB = TupleTools.getindices(szB, pB[2])
    sc = TupleTools.getindices(szA, pA[2])
    soA1 = prod(soA)
    soB1 = prod(soB)
    sc1 = prod(sc)
    pC = invperm(linearize(pAB))
    C̃ = reshape(PermutedDimsArray(C, pC), (soA1, soB1))

    if conjA && conjB
        Ã = reshape(PermutedDimsArray(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(PermutedDimsArray(B, linearize(reverse(pB))), (soB1, sc1))
        C̃ = mul!(C̃, adjoint(Ã), adjoint(B̃), α, β)
    elseif conjA
        Ã = reshape(PermutedDimsArray(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(PermutedDimsArray(B, linearize(pB)), (sc1, soB1))
        C̃ = mul!(C̃, adjoint(Ã), B̃, α, β)
    elseif conjB
        Ã = reshape(PermutedDimsArray(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(PermutedDimsArray(B, linearize(reverse(pB))), (soB1, sc1))
        C̃ = mul!(C̃, Ã, adjoint(B̃), α, β)
    else
        Ã = reshape(PermutedDimsArray(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(PermutedDimsArray(B, linearize(pB)), (sc1, soB1))
        C̃ = mul!(C̃, Ã, B̃, α, β)
    end
    return C
end
function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::BaseCopy)
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    szA = size(A)
    szB = size(B)
    soA = TupleTools.getindices(szA, pA[1])
    soB = TupleTools.getindices(szB, pB[2])
    sc = TupleTools.getindices(szA, pA[2])
    soA1 = prod(soA)
    soB1 = prod(soB)
    sc1 = prod(sc)

    if conjA && conjB
        Ã = reshape(permutedims(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(permutedims(B, linearize(reverse(pB))), (soB1, sc1))
        ÃB̃ = reshape(adjoint(Ã) * adjoint(B̃), (soA..., soB...))
    elseif conjA
        Ã = reshape(permutedims(A, linearize(reverse(pA))), (sc1, soA1))
        B̃ = reshape(permutedims(B, linearize(pB)), (sc1, soB1))
        ÃB̃ = reshape(adjoint(Ã) * B̃, (soA..., soB...))
    elseif conjB
        Ã = reshape(permutedims(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(permutedims(B, linearize(reverse(pB))), (soB1, sc1))
        ÃB̃ = reshape(Ã * adjoint(B̃), (soA..., soB...))
    else
        Ã = reshape(permutedims(A, linearize(pA)), (soA1, sc1))
        B̃ = reshape(permutedims(B, linearize(pB)), (sc1, soB1))
        ÃB̃ = reshape(Ã * B̃, (soA..., soB...))
    end
    if istrivialpermutation(linearize(pAB))
        pÃB̃ = ÃB̃
    else
        pÃB̃ = permutedims(ÃB̃, linearize(pAB))
    end
    if iszero(β)
        C .= α .* pÃB̃
    else
        C .= β .* C .+ α .* pÃB̃
    end
    return C
end

# ------------------------------------------------------------------------------------------
# Argument Checking: can be used by backends to check the validity of the arguments
# ------------------------------------------------------------------------------------------

"""
    argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)

Check that `C` has `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)
    return ndims(C) == numind(pC) && isperm(linearize(pC)) ||
           throw(IndexError("invalid permutation of length $(ndims(C)): $pC"))
end

"""
    argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have `numind(pA)` indices and that `pA` constitutes a valid permutation.
"""
function argcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)
    ndims(C) == ndims(A) || throw(IndexError("non-matching number of dimensions"))
    argcheck_index2tuple(A, pA)
    return nothing
end

"""
    argcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple)

Check that the partial trace of `A` over indices `q` and with permutation of the remaining
indices `p` is compatible with output `C`.
"""
function argcheck_tensortrace(C::AbstractArray, A::AbstractArray, p::Index2Tuple,
                              q::Index2Tuple)
    ndims(C) == numind(p) ||
        throw(IndexError("invalid selection of length $(ndims(C)): $p"))
    2 * numin(q) == 2 * numout(q) == ndims(A) - ndims(C) ||
        throw(IndexError("invalid number of trace dimensions"))
    return nothing
end

"""
    argcheck_tensorcontract(C::AbstractArray, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple, pAB::Index2Tuple)

Check that `C`, `A` and `pA`, and `B` and `pB` and `pAB` have compatible indices and number
of dimensions.
"""
function argcheck_tensorcontract(C::AbstractArray,
                                 A::AbstractArray, pA::Index2Tuple,
                                 B::AbstractArray, pB::Index2Tuple,
                                 pAB::Index2Tuple)
    argcheck_index2tuple(C, pAB)
    argcheck_index2tuple(A, pA)
    argcheck_index2tuple(B, pB)
    numout(pA) + numin(pB) == ndims(C) ||
        throw(IndexError("non-matching output indices in contraction"))
    numin(pA) == numout(pB) ||
        throw(IndexError("non-matching input indices in contraction"))
    return nothing
end

"""
    dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have compatible sizes for the addition specified by `pA`.
"""
function dimcheck_tensoradd(C::AbstractArray, A::AbstractArray, pA::Index2Tuple)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, linearize(pA)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray, A::AbstractArray,
                            p::Index2Tuple, q::Index2Tuple)

Check that `C` and `A` have compatible sizes for the trace and addition specified by `p` and
`q`.
"""
function dimcheck_tensortrace(C::AbstractArray, A::AbstractArray,
                              p::Index2Tuple, q::Index2Tuple)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, q[1]) == TupleTools.getindices(szA, q[2]) ||
        throw(DimensionMismatch("non-matching sizes in traced dimensions"))
    TupleTools.getindices(szA, linearize(p)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray,
                            A::AbstractArray, pA::Index2Tuple,
                            B::AbstractArray, pB::Index2Tuple,
                            pAB::Index2Tuple)

Check that `C`, `A` and `B` have compatible sizes for the contraction specified by `pA`,
`pB` and `pAB`.
"""
function dimcheck_tensorcontract(C::AbstractArray,
                                 A::AbstractArray, pA::Index2Tuple,
                                 B::AbstractArray, pB::Index2Tuple,
                                 pAB::Index2Tuple)
    szA, szB, szC = size(A), size(B), size(C)
    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    szAB = (TupleTools.getindices(szA, pA[1])..., TupleTools.getindices(szB, pB[2])...)
    TupleTools.getindices(szAB, linearize(pAB)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

#-------------------------------------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------------------------------------

flag2op(flag::Bool) = flag ? conj : identity
