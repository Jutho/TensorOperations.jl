tensorscalar(C::AbstractArray) = ndims(C) == 0 ? C[] : throw(DimensionMismatch())

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(A::AbstractArray, iA, conjA::Symbol,
                           B::AbstractArray, iB, conjB::Symbol, label)
    size(A, iA) == size(B, iB) ||
        throw(DimensionMismatch("Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
    return nothing
end

# TODO
# add check for stridedness of abstract arrays and add a pure implementation as fallback

const StridedNative = Backend{:StridedNative}
const StridedBLAS = Backend{:StridedBLAS}

function tensoradd!(C::AbstractArray,
                    A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                    α::Number, β::Number)
    return tensoradd!(C, A, pA, conjA, α, β, StridedNative())
end

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Symbol,
                      α::Number, β::Number)
    return tensortrace!(C, A, p, q, conjA, α, β, StridedNative())
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
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
                    A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                    α::Number, β::Number, backend::Union{StridedNative,StridedBLAS})
    tensoradd!(StridedView(C), StridedView(A), pA, conjA, α, β, backend)
    return C
end

function tensortrace!(C::AbstractArray,
                      A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Symbol,
                      α::Number, β::Number, backend::Union{StridedNative,StridedBLAS})
    tensortrace!(StridedView(C), StridedView(A), p, q, conjA, α, β, backend)
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::StridedBLAS)
    tensorcontract!(StridedView(C),
                    StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB,
                    pAB, α, β, StridedBLAS())
    return C
end

function tensorcontract!(C::AbstractArray,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple,
                         α::Number, β::Number, ::StridedNative)
    tensorcontract!(StridedView(C),
                    StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB,
                    pAB, α, β, StridedNative())
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

function flag2op(flag::Symbol)
    op = flag == :N ? identity :
         flag == :C ? conj :
         throw(ArgumentError("unknown conjugation flag $flag"))
    return op
end
