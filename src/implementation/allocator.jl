function TOC.tensoralloc(::Backend, args...)
    return tensor_from_structure(tensorstructure(args...)...)
end

function TOC.tensoralloctemp(backend::Backend, args...)
    return TOC.tensoralloc(backend, args...)
end

TOC.tensorfree!(::Backend, C) = nothing

# ---------------------------------------------------------------------------------------- #
# Interface for custom types
# ---------------------------------------------------------------------------------------- #

"""
    tensorstructure(TC, pC, A, conjA)
    tensorstructure(TC, pC, A, iA, conjA, B, iB, conjB)

Obtain the information needed to construct a new tensor with indices `pC`, based on the
indices `iA` (`iB`) of `opA(A)` (`opB(B)`). The operation `opA` (`opB`) acts as `conj` if
`conjA` (`conjB`) equals `:C` or as the identity if `conjA` (`conjB`) equals `:N`.
"""
function tensorstructure end

"""
    tensor_from_structure(T, structure)
    
Create a tensor based on a given structure.
"""
function tensor_from_structure end

# ---------------------------------------------------------------------------------------- #
# AbstractArray implementation
# ---------------------------------------------------------------------------------------- #

tensorop(args...) = +(*(args...), *(args...))
promote_contract(args...) = Base.promote_op(tensorop, args...)
promote_add(args...) = Base.promote_op(+, args...)

# output_scalartype() = Base.promote_op(tensorop, scalartype(A), scalartype(α))
# output_scalartype(A, B, α) = Base.promote_op(tensorop, scalartype(A), scalartype(B), scalartype(α))

tensorstructure(A::AbstractArray) = (typeof(A), size(A))
function tensorstructure(TC, pC, A::AbstractArray, conjA)
    sz = map(n -> size(A, n), linearize(pC))
    TType = Array{TC, length(sz)}
    return TType, sz
end

function tensorstructure(TC, pC, A::AbstractArray, iA::IndexTuple, conjA, B::AbstractArray,
                         iB::IndexTuple, conjB)
    sz = let lA = length(iA)
        map(linearize(pC)) do n
            if n <= lA
                return size(A, iA[n])
            else
                return size(B, iB[n - lA])
            end
        end
    end
    
    TType = Array{TC, length(sz)}
    return TType, sz
end

function tensor_from_structure(TType::Type{<:AbstractArray}, structure)
    A = similar(TType, structure)
    isbitstype(eltype(TType)) || fill!(A, zero(scalartype(TType)))
    return A
end
