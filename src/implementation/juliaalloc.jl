struct JuliaAllocator <: Backend end

function TOC.tensoralloc(::JuliaAllocator, args...)
    return tensor_from_structure(tensorstructure(args...)...)
end

function TOC.tensoralloctemp(backend::JuliaAllocator, args...)
    return TOC.tensoralloc(backend, args...)
end

TOC.tensorfree!(::JuliaAllocator, C) = nothing

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

tensorstructure(A::AbstractArray{T,N}) where {T,N} = (A, scalartype(T), size(A))
tensorstructure(TC, pC, A::AbstractArray, _) = A, TC, map(n -> size(A, n), linearize(pC))

function tensorstructure(TC, pC, A::AbstractArray, iA::IndexTuple, _, B::AbstractArray, iB::IndexTuple, _)
    sz = let lA = length(iA)
        map(linearize(pC)) do n
            if n <= lA
                return size(A, iA[n])
            else
                return size(B, iB[n - lA])
            end
        end
    end
    return A, TC, sz
end

function tensor_from_structure(A::AbstractArray, T, sz)
    if isbitstype(T)
        return similar(A, T, sz)
    else
        return fill!(similar(A, T, sz), zero(T))
    end
end