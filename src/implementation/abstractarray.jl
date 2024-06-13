tensorscalar(C::AbstractArray) = ndims(C) == 0 ? C[] : throw(DimensionMismatch())

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(A::AbstractArray, iA, conjA::Bool,
                           B::AbstractArray, iB, conjB::Bool, label)
    size(A, iA) == size(B, iB) ||
        throw(DimensionMismatch("Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
    return nothing
end

const StridedNative = Backend{:StridedNative}
const StridedBLAS = Backend{:StridedBLAS}

function tensoradd!(C::StridedArray,
                    A::StridedArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number)
    return tensoradd!(C, A, pA, conjA, α, β, StridedNative())
end

function tensortrace!(C::StridedArray,
                      A::StridedArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number)
    return tensortrace!(C, A, p, q, conjA, α, β, StridedNative())
end

function tensorcontract!(C::StridedArray,
                         A::StridedArray, pA::Index2Tuple, conjA::Bool,
                         B::StridedArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number)
    if eltype(C) <: LinearAlgebra.BlasFloat && !isa(B, Diagonal) && !isa(A, Diagonal)
        return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, StridedBLAS())
    else
        return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, StridedNative())
    end
end

#-------------------------------------------------------------------------------------------
# Implementation based on StridedViews
#-------------------------------------------------------------------------------------------

function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                    α::Number, β::Number, backend::Union{StridedNative,StridedBLAS})
    tensoradd!(StridedView(C), StridedView(A), pA, conjA, α, β, backend)
    return C
end

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number, backend::Union{StridedNative,StridedBLAS})
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
                    α::Number, β::Number)
    argcheck_tensoradd(C, pC, A)
    dimcheck_tensoradd(C, pC, A)

    # can we assume that C is mutable?
    # is there more functionality in base that we can use?
    if conjA
        C .= β .* C .+ α .* conj.(PermutedDimsArray(A, linearize(pA)))
    else
        C .= β .* C .+ α .* PermutedDimsArray(A, linearize(pA))
    end
    return C
end

# For now I am giving up on writing a generic tensortrace! that works for all AbstractArray types...

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                      α::Number, β::Number, backend::Union{StridedNative,StridedBLAS})
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    szA = size(A)
    so = TupleTools.getindices(szA, linearize(p))
    st = prod(TupleTools.getindices(szA, q[1]))
    A = reshape(PermutedDimsArray(A, (linearize(p)..., linearize(q)...)), (so..., st * st))
    if conjA
        C .= β .* C .+ α .* conj.(view(A, :, diagind(st, st)))
    else
        C .= β .* C .+ α .* view(A, :, diagind(st, st))
    end
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                         B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number)
    argcheck_tensorcontract(C, pC, A, pA, B, pB)
    dimcheck_tensorcontract(C, pC, A, pA, B, pB)

    szA = size(A)
    szB = size(B)
    soA = TupleTools.getindices(szA, pA[1])
    soB = TupleTools.getindices(szA, pB[2])
    sc = TupleTools.getindices(szA, pA[2])

    if conjA && conjB
        A′ = reshape(permutedims(A, linearize(reverse(pA))), (sc, soA))
        B′ = reshape(permutedims(B, linearize(reverse(pB))), (soB, sc))
        C′ = adjoint(A′) * adjoint(B′)
    elseif conjA
        A′ = reshape(permutedims(A, linearize(reverse(pA))), (sc, soA))
        B′ = reshape(permutedims(B, linearize(pB)), (sc, soB))
        C′ = adjoint(A′) * B′
    elseif conjB
        A′ = reshape(permutedims(A, linearize(pA)), (soA, sc))
        B′ = reshape(permutedims(B, linearize(reverse(pB))), (soB, sc))
        C′ = A′ * adjoint(B′)
    else
        A′ = reshape(permutedims(A, linearize(pA)), (soA, sc))
        B′ = reshape(permutedims(B, linearize(pB)), (sc, soB))
        C′ = A′ * B′
    end
    return tensoradd!(C, C′, pAB, false, α, β)
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
