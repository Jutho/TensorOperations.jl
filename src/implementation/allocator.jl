#===========================================================================================
    Generic implementation
===========================================================================================#

tensorop(args...) = +(*(args...), *(args...))
promote_contract(args...) = Base.promote_op(tensorop, args...)
promote_add(args...) = Base.promote_op(+, args...)

function tensoralloc_add(TC, A, pA, conjA, istemp=false)
    ttype = tensoradd_type(TC, A, pA, conjA)
    structure = tensoradd_structure(A, pA, conjA)
    return tensoralloc(ttype, structure, istemp)::ttype
end

function tensoralloc_contract(TC, pC, A, pA, conjA, B, pB, conjB,
                              istemp=false)
    ttype = tensorcontract_type(TC, pC, A, pA, conjA, B, pB, conjB)
    structure = tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)
    return tensoralloc(ttype, structure, istemp)::ttype
end

tensorfree!(C) = nothing

#===========================================================================================
    AbstractArray implementation
===========================================================================================#

tensorstructure(A::AbstractArray) = size(A)
tensorstructure(A::AbstractArray, iA::Int, conjA::Symbol) = size(A, iA)

function tensoradd_type(TC, A::AbstractArray, pA::Index2Tuple, conjA::Symbol)
    return Array{TC,sum(length.(pA))}
end

function tensoradd_structure(A::AbstractArray, pA::Index2Tuple, conjA::Symbol)
    return size.(Ref(A), linearize(pA))
end

function tensorcontract_type(TC, pC, A::AbstractArray, pA, conjA,
                             B::AbstractArray, pB, conjB)
    return Array{TC,sum(length.(pC))}
end

function tensorcontract_structure(pC::Index2Tuple,
                                  A::AbstractArray, pA::Index2Tuple, conjA,
                                  B::AbstractArray, pB::Index2Tuple, conjB)
    return let lA = length(pA[1])
        map(n -> n <= lA ? size(A, pA[1][n]) : size(B, pB[2][n - lA]), linearize(pC))
    end
end

function tensoralloc(ttype, structure, istemp=false)
    C = ttype(undef, structure)
    isbitstype(scalartype(ttype)) || fill!(C, zero(scalartype(ttype)))
    return C
end