# auxiliary/stridedarray.jl
#
# Simple auxiliary methods to interface with StridedArray from Julia Base.


"""
    numind(A)

Returns the number of indices of a tensor-like object `A`, i.e. for a multidimensional
array (`<:AbstractArray`) we have `numind(A) = ndims(A)`. Also works in type domain.
"""
numind(A::AbstractArray) = ndims(A)
numind(::Type{T}) where {T<:AbstractArray} = ndims(T)


checkindices(A::AbstractArray{<:Any, N}, IA::IndexTuple{N}) where {N} = true

"""
    similar_from_indices(::Type{T}, indices::NTuple{N,Int}, A, conjA=Val{:N}) where {T,N}

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes
corresponding to a selection of those of `op(A)`, where the selection is specified by
`indices` (which contains integer between `1` and `numind(A)`) and `op` is `conj` if
`conjA=Val{:C}` or does nothing if `conjA=Val{:N}` (default).
"""
function similar_from_indices(T::Type, indices::IndexTuple, A::AbstractArray, ::Type{<:Val}=Val{:N})
    return similar(A, T, map(n->size(A,n), indices))
end

similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A::AbstractArray, CA::Type{<:Val}) = similar_from_indices(T, (p1...,p2...), A, CA)

"""
    similar_from_indices(::Type{T}, indices::NTuple{N,Int}, A, B, conjA=Val{:N}, conjB={:N}) where {T,N}

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes
corresponding to a selection of those of `op(A)` and `op(B)` concatenated, where the
selection is specified by `indices` (which contains integers between `1` and
    `numind(A)+numind(B)` and `op` is `conj` if `conjA` or `conjB` equal `Val{:C}`
    or does nothing if `conjA` or `conjB` equal `Val{:N}` (default).
"""
function similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple, p1::IndexTuple, p2::IndexTuple, A::AbstractArray, B::AbstractArray, CA::Type{<:Val}, CB::Type{<:Val})
    odimsA = map(n->size(A,n), poA)
    odimsB = map(n->size(B,n), poB)
    odimsAB = (odimsA...,odimsB...)
    dimsC = map(n->odimsAB[n], (p1...,p2...))
    return similar(A, T, dimsC)
end

"""
    scalar(C)

Returns the single element of a tensor-like object with zero dimensions, i.e. if `numind(C)==0`.
"""
scalar(C::AbstractArray) = numind(C)==0 ? C[1] : throw(DimensionMismatch())
