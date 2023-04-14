module TensorOperationsTBLIS

using TensorOperationsCore
using TensorOperations
using TupleTools

isdefined(Base, :get_extension) ? (using TBLIS) : (using ..TBLIS)


function TensorOperationsCore.tensoradd!(::TBLISBackend, C::AbstractArray, A::AbstractArray,
                                         pA::Index2Tuple, conjA::Symbol, α::Number,
                                         β::Number)
    ndims(C) == ndims(A) || throw(DimensionMismatch("ndims(A) ≠ ndims(C)"))
    ndims(C) == length(pA[1]) + length(pA[2]) ||
        throw(IndexError("Invalid permutation of length $(ndims(C)): $pA"))
    
    # check dimensions
    szC = size(C)
    szA = size(A)
    szC == getindex.(Ref(szA), linearize(pA)) || throw(DimensionMismatch("incompatible sizes"))
    
    # convert to TBLIS tensors
    C_TT = TTensor{scalartype(C)}(C, β)
    A_TT = TTensor{scalartype(A)}(A, α)
    conjA === :C && conj!(A_TT)

    einA, einC = TensorOperations.add_labels(pA)

    TBLIS.add!(A_TT, C_TT, string(einA...), string(einC...))
    return C
end

function TensorOperationsCore.tensortrace!(::TBLISBackend, C::AbstractArray,
                                           pC::Index2Tuple,
                                           A::AbstractArray, pA::Index2Tuple, conjA::Symbol,
                                           α, β)
    NA, NC = ndims(A), ndims(C)
    NC == sum(length.(pC)) ||
        throw(IndexError("invalid selection of $NC out of $NA: $pC"))
    NA - NC == 2 * length(pA[1]) == 2 * length(pA[2]) ||
        throw(IndexError("invalid number of trace dimensions"))

    szC = size(C)
    szA = size(A)
    szC == getindex.(Ref(szA), linearize(pC)) ||
        throw(DimensionMismatch("incompatible open sizes"))
    getindex.(Ref(szA), pA[1]) == getindex.(Ref(szA), pA[2]) ||
        throw(DimensionMismatch("incompatible traced sizes"))
    
    C_TT = TTensor{scalartype(C)}(C, β)
    A_TT = TTensor{scalartype(A)}(A, α)
    conjA === :C && conj!(A_TT)

    einA, einC = splat(string).(TensorOperations.trace_labels(pC, pA[1], pA[2]))
    TBLIS.add!(A_TT, C_TT, einA, einC)
    return C
end

function TensorOperationsCore.tensorcontract!(::TBLISBackend, C::AbstractArray,
                                              pC::Index2Tuple,
                                              A::AbstractArray, pA::Index2Tuple, conjA,
                                              B::AbstractArray, pB::Index2Tuple, conjB,
                                              α, β)
    (length(pA[1]) + length(pA[2]) == ndims(A) && TupleTools.isperm(linearize(pA))) ||
        throw(IndexError("invalid permutation of A of length $(ndims(A)): $pA"))
    (length(pB[1]) + length(pB[2]) == ndims(B) && TupleTools.isperm(linearize(pB))) ||
        throw(IndexError("invalid permutation of B of length $(ndims(B)): $pB"))
    (length(pA[1]) + length(pB[2]) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (length(pC[1]) + length(pC[2]) == ndims(C) && TupleTools.isperm(linearize(pC))) ||
        throw(IndexError("invalid permutation of C of length $(ndims(C)): $pC"))

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

    # convert to TBLIS tensors
    C_TT = TTensor{scalartype(C)}(C, β)
    A_TT = TTensor{scalartype(A)}(A, α)
    B_TT = TTensor{scalartype(B)}(B)

    conjA === :C && conj!(A_TT)
    conjB === :C && conj!(B_TT)

    # convert to TBLIS idx
    einA, einB, einC = TensorOperations.contract_labels(pA, pB, pC)

    TBLIS.mul!(C_TT, A_TT, B_TT, string(einA...), string(einB...), string(einC...))
    return C
end

end