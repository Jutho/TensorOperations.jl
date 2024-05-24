# ------------------------------------------------------------------------------------------
# Generic implementation
# ------------------------------------------------------------------------------------------

tensorop(args...) = +(*(args...), *(args...))
"""
    promote_contract(args...)

Promote the scalar types of a tensor contraction to a common type.
"""
promote_contract(args...) = Base.promote_op(tensorop, args...)

"""
    promote_add(args...)

Promote the scalar types of a tensor addition to a common type.
"""
promote_add(args...) = Base.promote_op(+, args...)

"""
    tensoralloc_add(TC, A, pA, conjA, istemp=false, backend::Backend...)

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensoradd!(C, A, pA, conjA)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `backend` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_add(TC, A, pA::Index2Tuple, conjA::Bool, istemp::Bool=false,
                         backend::Backend...)
    ttype = tensoradd_type(TC, A, pA, conjA)
    structure = tensoradd_structure(A, pA, conjA)
    return tensoralloc(ttype, structure, istemp, backend...)::ttype
end

"""
    tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB, istemp=false, backend::Backend...)

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `backend` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_contract(TC,
                              A, pA::Index2Tuple, conjA::Bool,
                              B, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple, istemp::Bool=false, backend::Backend...)
    ttype = tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure, istemp, backend...)::ttype
end

# ------------------------------------------------------------------------------------------
# AbstractArray implementation
# ------------------------------------------------------------------------------------------

tensorstructure(A::AbstractArray) = size(A)
tensorstructure(A::AbstractArray, iA::Int, conjA::Bool) = size(A, iA)

function tensoradd_type(TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return Array{TC,sum(length.(pA))}
end

function tensoradd_structure(A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return size.(Ref(A), linearize(pA))
end

function tensorcontract_type(TC, A::AbstractArray, pA, conjA,
                             B::AbstractArray, pB, conjB, pAB, backend...)
    return Array{TC,sum(length.(pAB))}
end

function tensorcontract_structure(A::AbstractArray, pA, conjA,
                                  B::AbstractArray, pB, conjB, pAB, backend...)
    return let lA = length(pA[1])
        map(n -> n <= lA ? size(A, pA[1][n]) : size(B, pB[2][n - lA]), linearize(pAB))
    end
end

function tensoralloc(ttype, structure, istemp=false, backend::Backend...)
    C = ttype(undef, structure)
    # fix an issue with undefined references for strided arrays
    if !isbitstype(scalartype(ttype))
        C = zerovector!!(C)
    end
    return C
end
