tensorscalar(C::AbstractArray) = ndims(C) == 0 ? C[] : throw(DimensionMismatch())

tensorcost(C::AbstractArray, i) = size(C, i)

function checkcontractible(A::AbstractArray, iA, conjA::Symbol,
                           B::AbstractArray, iB, conjB::Symbol, label)
    size(A, iA) == size(B, iB) ||
        throw(IndexError("Nonmatching dimensions for $label: $(size(A, iA)) != $(size(B, iB))"))
    return nothing
end

# TODO
# add check for stridedness of abstract arrays and add a pure implementation as fallback

const StridedNative = Backend{:StridedNative}
const StridedBLAS = Backend{:StridedBLAS}

function tensoradd!(C::AbstractArray, pC::Index2Tuple,
                    A::AbstractArray, conjA::Symbol,
                    α, β)
    return tensoradd!(C, pC, A, conjA, α, β, StridedNative())
end

function tensortrace!(C::AbstractArray, pC::Index2Tuple,
                      A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                      α, β)
    return tensortrace!(C, pC, A, pA, conjA, α, β, StridedNative())
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         α, β)
    if eltype(C) <: LinearAlgebra.BlasFloat && !isa(B, Diagonal) && !isa(A, Diagonal)
        return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, StridedBLAS())
    else
        return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β, StridedNative())
    end
end

#-------------------------------------------------------------------------------------------
# Implementation based on StridedViews
#-------------------------------------------------------------------------------------------

function tensoradd!(C::AbstractArray, pC::Index2Tuple,
                    A::AbstractArray, conjA::Symbol,
                    α, β, backend::Union{StridedNative,StridedBLAS})
    tensoradd!(StridedView(C), pC, StridedView(A), conjA, α, β, backend)
    return C
end

function tensortrace!(C::AbstractArray, pC::Index2Tuple,
                      A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                      α, β, backend::Union{StridedNative,StridedBLAS})
    tensortrace!(StridedView(C), pC, StridedView(A), pA, conjA, α, β, backend)
    return C
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedBLAS)
    tensorcontract!(StridedView(C), pC, StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB, α, β, StridedBLAS())
    return C
end

function tensorcontract!(C::AbstractArray, pC::Index2Tuple,
                         A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractArray, pB::Index2Tuple, conjB::Symbol,
                         α, β, ::StridedNative)
    tensorcontract!(StridedView(C), pC, StridedView(A), pA, conjA,
                    StridedView(B), pB, conjB, α, β, StridedNative())
    return C
end

#===========================================================================================
    Argument Checking: can be used by backends to check the validity of the arguments
===========================================================================================#

"""
    argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)

Check that `C` has `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_index2tuple(C::AbstractArray, pC::Index2Tuple)
    return ndims(C) == numind(pC) && isperm(linearize(pC)) ||
           throw(IndexError("invalid permutation of length $(ndims(C)): $pC"))
end

"""
    argcheck_tensoradd(C::AbstractArray, pC::Index2Tuple, A::AbstractArray)

Check that `C` and `A` have `numind(pC)` indices and that `pC` constitutes a valid permutation.
"""
function argcheck_tensoradd(C::AbstractArray, pC::Index2Tuple, A::AbstractArray)
    ndims(C) == ndims(A) || throw(IndexError("non-matching number of dimensions"))
    argcheck_index2tuple(C, pC)
    return nothing
end

"""
    argcheck_tensortrace(C::AbstractArray, pC::Index2Tuple, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `pC` have compatible indices and number of dimensions with the trace of
`A` over indices `pA`.
"""
function argcheck_tensortrace(C::AbstractArray, pC::Index2Tuple, A::AbstractArray,
                              pA::Index2Tuple)
    ndims(C) == numind(pC) ||
        throw(IndexError("invalid selection of length $(ndims(C)): $pC"))
    2 * numin(pA) == 2 * numout(pA) == ndims(A) - ndims(C) ||
        throw(IndexError("invalid number of trace dimensions"))
    return nothing
end

"""
    argcheck_tensorcontract(C::AbstractArray, pC::Index2Tuple, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple)

Check that `C` and `pC`, `A` and `pA`, and `B` and `pB` have compatible indices and number
of dimensions.
"""
function argcheck_tensorcontract(C::AbstractArray, pC::Index2Tuple,
                                 A::AbstractArray, pA::Index2Tuple,
                                 B::AbstractArray, pB::Index2Tuple)
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
    dimcheck_tensoradd(C::AbstractArray, pC::Index2Tuple, A::AbstractArray)

Check that `C` and `A` have compatible sizes for the addition specified by `pC`.
"""
function dimcheck_tensoradd(C::AbstractArray, pC::Index2Tuple, A::AbstractArray)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, linearize(pC)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray, pC::Index2Tuple, A::AbstractArray, pA::Index2Tuple)

Check that `C` and `A` have compatible sizes for the trace and addition specified by `pC` and `pA`.
"""
function dimcheck_tensortrace(C::AbstractArray, pC::Index2Tuple, A::AbstractArray,
                              pA::Index2Tuple)
    szA, szC = size(A), size(C)
    TupleTools.getindices(szA, pA[1]) == TupleTools.getindices(szA, pA[2]) ||
        throw(DimensionMismatch("non-matching sizes in traced dimensions"))
    TupleTools.getindices(szA, linearize(pC)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

"""
    dimcheck_tensorcontract(C::AbstractArray, pC::Index2Tuple, A::AbstractArray, pA::Index2Tuple, B::AbstractArray, pB::Index2Tuple)

Check that `C`, `A` and `B` have compatible sizes for the contraction specified by `pC`, `pA` and `pB`.
"""
function dimcheck_tensorcontract(C::AbstractArray, pC::Index2Tuple,
                                 A::AbstractArray, pA::Index2Tuple,
                                 B::AbstractArray, pB::Index2Tuple)
    szA, szB, szC = size(A), size(B), size(C)
    TupleTools.getindices(szA, pA[2]) == TupleTools.getindices(szB, pB[1]) ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    szAB = (TupleTools.getindices(szA, pA[1])..., TupleTools.getindices(szB, pB[2])...)
    TupleTools.getindices(szAB, linearize(pC)) == szC ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    return nothing
end

#-------------------------------------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------------------------------------
function flag2op(flag::Symbol)
    op = flag == :N ? identity :
         flag == :C ? conj :
         throw(ArgumentError("unknown conjuagation flag $flag"))
    return op
end
