const PRECOMPILE_ELTYPES = (Float64, ComplexF64)

# tensoradd!
# ----------
const PRECOMPILE_ADD_NDIMS = 5

for T in PRECOMPILE_ELTYPES
    for N in 0:PRECOMPILE_ADD_NDIMS
        C = Array{T,N}
        A = Array{T,N}
        pA = Index2Tuple{N,0}

        precompile(tensoradd!, (C, A, pA, Bool, One, Zero))
        precompile(tensoradd!, (C, A, pA, Bool, T, Zero))
        precompile(tensoradd!, (C, A, pA, Bool, T, T))

        precompile(tensoralloc_add, (T, A, pA, Bool, Val{true}))
        precompile(tensoralloc_add, (T, A, pA, Bool, Val{false}))
    end
end

# tensortrace!
# ------------
const PRECOMPILE_TRACE_NDIMS = (4, 2)

for T in PRECOMPILE_ELTYPES
    for N1 in 0:PRECOMPILE_TRACE_NDIMS[1], N2 in 0:PRECOMPILE_TRACE_NDIMS[2]
        C = Array{T,N1}
        A = Array{T,N1 + 2N2}
        p = Index2Tuple{N1,0}
        q = Index2Tuple{N2,N2}

        precompile(tensortrace!, (C, A, p, q, Bool, One, Zero))
        precompile(tensortrace!, (C, A, p, q, Bool, T, Zero))
        precompile(tensortrace!, (C, A, p, q, Bool, T, T))

        # allocation re-uses tensoralloc_add
    end
end

# tensorcontract!
# ---------------
const PRECOMPILE_CONTRACT_NDIMS = (3, 2, 3)

for T in PRECOMPILE_ELTYPES
    for N1 in 0:PRECOMPILE_CONTRACT_NDIMS[1], N2 in 0:PRECOMPILE_CONTRACT_NDIMS[2],
        N3 in 0:PRECOMPILE_CONTRACT_NDIMS[3]

        NA = N1 + N2
        NB = N2 + N3
        NC = N1 + N3
        C, A, B = Array{T,NC}, Array{T,NA}, Array{T,NB}
        pA = Index2Tuple{N1,N2}
        pB = Index2Tuple{N2,N3}
        pAB = Index2Tuple{NC,0}

        precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, One, Zero))
        precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, T, Zero))
        precompile(tensorcontract!, (C, A, pA, Bool, B, pB, Bool, pAB, T, T))

        precompile(tensoralloc_contract, (T, A, pA, Bool, B, pB, Bool, pAB, Val{true}))
        precompile(tensoralloc_contract, (T, A, pA, Bool, B, pB, Bool, pAB, Val{false}))
    end
end
