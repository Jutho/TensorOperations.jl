"""
    struct IndexError{<:AbstractString} <: Exception
    
exception type for reporting errors in the index specification.
"""
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

"""
    IndexTuple{N}

A specification of `N` selected tensor indices, denoted by their position.
"""
const IndexTuple{N} = NTuple{N,Int}

"""
    Index2Tuple{N₁,N₂}

A specification of a permutation of `N₁ + N₂` indices that are partitioned into `N₁` left
and `N₂` right indices.
"""
const Index2Tuple{N₁,N₂} = Tuple{IndexTuple{N₁},IndexTuple{N₂}}

linearize(p::Index2Tuple) = (p[1]..., p[2]...)
numout(p::Index2Tuple) = length(p[1])
numin(p::Index2Tuple) = length(p[2])
numind(p::Index2Tuple) = numout(p) + numin(p)

#===========================================================================================
    Argument Checking
===========================================================================================#

"""
    argcheck_index2tuple(C, pC::Index2Tuple)

Check that `C` has `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_index2tuple(C, pC::Index2Tuple)
    return ndims(C) == numind(pC) && isperm(linearize(pC)) ||
           throw(IndexError("invalid permutation of length $(ndims(C)): $pC"))
end

"""
    argcheck_tensoradd(C, pC::Index2Tuple, A)

Check that `C` and `A` have `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_tensoradd(C, pC::Index2Tuple, A)
    ndims(C) == ndims(A) || throw(IndexError("non-matching number of dimensions"))
    argcheck_index2tuple(C, pC)
    return nothing
end

"""
    argcheck_tensorcontract(C, pC::Index2Tuple, A, pA::Index2Tuple, B, pB::Index2Tuple)

Check that `C` and `pC`, `A` and `pA`, and `B` and `pB` have compatible indices and number
of dimensions.
"""
function argcheck_tensorcontract(C, pC::Index2Tuple, A, pA::Index2Tuple, B, pB::Index2Tuple)
    argcheck_index2tuple(C, pC)
    argcheck_index2tuple(A, pA)
    argcheck_index2tuple(B, pB)
    numout(pA) + numin(pB) == ndims(C) ||
        throw(IndexError("non-matching output indices in contraction"))
    numin(pA) == numout(pB) ||
        throw(IndexError("non-matching input indices in contraction"))
    return nothing
end

"""
    dimcheck_tensorcontract(C, pC::Index2Tuple, A, pA::Index2Tuple, B, pB::Index2Tuple)

Check that `C`, `A` and `B` have compatible sizes for the contraction specified by `pC`, `pA` and `pB`.
"""
function dimcheck_tensorcontract(C, pC::Index2Tuple, A, pA::Index2Tuple, B, pB::Index2Tuple)
    szA, szB, szC = size(A), size(B), size(C)
    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    szAB = (TupleTools.getindices(szA, pA[1])..., TupleTools.getindices(szB, pB[2])...)
    TupleTools.getindices(szAB, linearize(pC)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    argcheck_tensortrace(C, pC, A, pA)

Check that `C` and `pC` have compatible indices and number of dimensions with the trace of
`A` over indices `pA`.
"""
function argcheck_tensortrace(C, pC::Index2Tuple, A, pA::Index2Tuple)
    ndims(C) == numind(pC) ||
        throw(IndexError("invalid selection of length $(ndims(C)): $pC"))
    2 * numin(pA) == 2 * numout(pA) == ndims(A) - ndims(C) ||
        throw(IndexError("invalid number of trace dimensions"))
    return nothing
end
