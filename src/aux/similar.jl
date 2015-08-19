numind(A::StridedArray) = ndims(A)
numind{T<:StridedArray}(::Type{T}) = ndims(T)

function similar_from_indices{T,CA}(::Type{T}, indices, A::StridedArray, ::Type{Val{CA}}=Val{:N})
    dims = size(A)
    return Array{T}(dims[indices])
end

function similar_from_indices{T,CA,CB}(::Type{T}, indices, A::StridedArray, B::StridedArray, ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N})
    dims = tuple(size(A)...,size(B)...)
    return Array{T}(dims[indices])
end
