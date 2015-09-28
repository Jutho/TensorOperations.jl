# aux/stridedarray.jl
#
# Simple auxiliary methods to interface with StridedArray from Julia Base.

numind(A::StridedArray) = ndims(A)
numind{T<:StridedArray}(::Type{T}) = ndims(T)

function similar_from_indices{T,CA}(::Type{T}, indices, A::StridedArray, ::Type{Val{CA}}=Val{:N})
    dims = size(A)
    return similar(A,T,dims[indices])
end

function similar_from_indices{T,CA,CB}(::Type{T}, indices, A::StridedArray, B::StridedArray, ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N})
    dims = tuple(size(A)...,size(B)...)
    return similar(A,T,dims[indices])
end

scalar(C::StridedArray) = length(C)==1 ? C[1] : throw(DimensionMismatch())
