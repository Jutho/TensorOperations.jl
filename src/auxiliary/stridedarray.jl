# auxiliary/stridedarray.jl
#
# Simple auxiliary methods to interface with StridedArray from Julia Base.


"""`numind(A)`

Returns the number of indices of a tensor-like object `A`, i.e. for a multidimensional array (`<:AbstractArray`) we have `numind(A) = ndims(A)`. Also works in type domain.
"""
numind(A::AbstractArray) = ndims(A)
numind(::Type{T}) where {T<:AbstractArray} = ndims(T)

"""`similar_from_indices(T, indices, A, conjA=Val{:N})`

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)`, where the selection is specified by `indices` (which contains integer between `1` and `numind(A)`) and `op` is `conj` if `conjA=Val{:C}` or does nothing if `conjA=Val{:N}` (default).
"""
function similar_from_indices(::Type{T}, indices, A::StridedArray, ::Type{Val{CA}}=Val{:N}) where {T,CA}
    dims = size(A)
    return similar(A,T,dims[indices])
end

"""`similar_from_indices(T, indices, A, B, conjA=Val{:N}, conjB={:N})`

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes corresponding to a selection of those of `op(A)` and `op(B)` concatenated, where the selection is specified by `indices` (which contains integers between `1` and `numind(A)+numind(B)` and `op` is `conj` if `conjA` or `conjB` equal `Val{:C}` or does nothing if `conjA` or `conjB` equal `Val{:N}` (default).
"""
function similar_from_indices(::Type{T}, indices, A::StridedArray, B::StridedArray, ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N}) where {T,CA,CB}
    dims = tuple(size(A)...,size(B)...)
    return similar(A,T,dims[indices])
end

"""`scalar(C)`

Returns the single element of a tensor-like object with zero dimensions, i.e. if `numind(C)==0`.
"""
scalar(C::StridedArray) = numind(C)==0 ? C[1] : throw(DimensionMismatch())
