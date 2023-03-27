module TensorOperationsTBLIS

using TensorOperationsCore
using TBLIS

export TBLISBackend
struct TBLISBackend <: TensorOperationsCore.Backend end

function TensorOperationsCore.tensoradd!(::TBLISBackend, C::AbstractArray, A::AbstractArray,
     pA::Index2Tuple, conjA::Symbol, α::Number, β::Number)
    # convert to TBLIS tensors
    C_TT = TTensor{scalartype(C)}(C, β)
    A_TT = TTensor{scalartype(A)}(A, α)
    conjA === :C && conj!(A_TT)
    
    # convert to TBLIS idx
    pA_lin = linearize(pA)
    idx_A = 'a' .+ (1:ndims(A)) .- 1
    idx_C = getindex.(Ref(idx_A), pA_lin)
    
    TBLIS.add!(A_TT, C_TT, string(idx_A...), string(idx_C...))
    return C
end

function tblisidx_contract(pA::Index2Tuple, pB::Index2Tuple, pC::Index2Tuple)
    pC_lin = linearize(pC)
    
    idx_A = Vector{Char}(undef, length(pA[1]) + length(pA[2]))
    idx_B = Vector{Char}(undef, length(pB[1]) + length(pB[2]))
    
    # open idx
    for i = 1:length(pA[1])
        idx_A[pA[1][i]] = i + 'a' - 1
    end
    for i = 1:length(pB[2])
        idx_B[pB[2][i]] = i + length(pA[1]) + 'a' - 1
    end
    
    # contracted idx
    for i = 1:length(pA[2])
        idx_A[pA[2][i]] = i + 'A' - 1
        idx_B[pB[1][i]] = i + 'A' - 1
    end
    
    idx_A = string(idx_A...)
    idx_B = string(idx_B...)
    idx_C = string((pC_lin .+ ('a' - 1))...)
    
    return idx_A, idx_B, idx_C
end

function TensorOperationsCore.tensorcontract!(::TBLISBackend, C::AbstractArray,
                                              pC::Index2Tuple,
                                              A::AbstractArray, pA::Index2Tuple, conjA,
                                              B::AbstractArray, pB::Index2Tuple, conjB,
                                              α, β)
    # convert to TBLIS tensors
    C_TT = TTensor{scalartype(C)}(C, β)
    A_TT = TTensor{scalartype(A)}(A, α)
    B_TT = TTensor{scalartype(B)}(B)

    conjA === :C && conj!(A_TT)
    conjB === :C && conj!(B_TT)
    
    # convert to TBLIS idx
    idx_A, idx_B, idx_C = tblisidx_contract(pA, pB, pC)

    TBLIS.mul!(C_TT, A_TT, B_TT, idx_A, idx_B, idx_C)
    return C
end

end