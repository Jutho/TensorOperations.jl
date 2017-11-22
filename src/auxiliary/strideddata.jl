# auxiliary/strideddata.jl
#
# Wrapper to group data (as vector indicating a memory region), a starting offset
# inside the region and a collection of strides to interpret this memory as a
# multidimensional array.

struct StridedData{N,T,C}
    data::Vector{T}
    strides::NTuple{N,Int}
    start::Int
end

NormalStridedData{N,T} =  StridedData{N,T,:N}
ConjugatedStridedData{N,T} =  StridedData{N,T,:C}

import Base.StridedReshapedArray

StridedSubArray{T,N,A<:Union{DenseArray{T},StridedReshapedArray{T}},I<:Tuple{Vararg{Union{Base.RangeIndex, Base.AbstractCartesianIndex}}}} =  SubArray{T,N,A,I}

StridedData(a::Array{T}, strides::IndexTuple{N} = strides(a), ::Type{Val{C}} = Val{:N}; offset::Int = 0) where {N,T,C} =
    StridedData{N,T,C}(vec(a), strides, 1+offset)
StridedData(a::StridedReshapedArray{T}, strides::IndexTuple{N} = strides(a), ::Type{Val{C}} = Val{:N}; offset::Int = 0) where {N,T,C} =
    StridedData(a.parent, strides, Val{C}; offset = offset)
StridedData(a::StridedSubArray{T}, strides::IndexTuple{N} = strides(a), ::Type{Val{C}} = Val{:N}; offset::Int = 0) where {N,T,C} =
    StridedData(a.parent, strides, Val{C}; offset = offset+Base.first_index(a)-1)

Base.getindex(a::NormalStridedData, i) = a.data[i]
Base.getindex(a::ConjugatedStridedData, i) = conj(a.data[i])

Base.setindex!(a::NormalStridedData, v, i) = (@inbounds a.data[i] = v)
Base.setindex!(a::ConjugatedStridedData, v, i) = (@inbounds a.data[i] = conj(v))

# set dimensions dims[d]==1 for all d where a.strides[d] == 0.
@generated function _filterdims(dims::IndexTuple{N}, a::StridedData{N}) where {N}
    meta = Expr(:meta,:inline)
    ex = Expr(:tuple,[:(a.strides[$d]==0 ? 1 : dims[$d]) for d=1:N]...)
    Expr(:block,meta,ex)
end

# initial scaling of a block specified by dims
_scale!(C::StridedData{N}, β::One, dims::IndexTuple{N}, offset::Int=0) where {N} = C

@generated function _scale!(C::StridedData{N}, β::Zero, dims::IndexTuple{N}, offset::Int=0) where {N}
    meta = Expr(:meta,:inline)
    quote
        $meta
        dims = _filterdims(dims,C)
        startC = C.start+offset
        stridesC = C.strides
        @stridedloops($N, dims, indC, startC, stridesC, @inbounds C[indC] = false)
        return C
    end
end

@generated function _scale!(C::StridedData{N}, β::Number, dims::IndexTuple{N}, offset::Int=0) where {N}
    meta = Expr(:meta,:inline)
    quote
        $meta
        dims = _filterdims(dims,C)
        startC = C.start+offset
        stridesC = C.strides
        @stridedloops($N, dims, indC, startC, stridesC, @inbounds C[indC] *= β)
        return C
    end
end
