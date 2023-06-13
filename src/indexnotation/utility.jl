"""
    struct IndexError{<:AbstractString} <: Exception
    
exception type for reporting errors in the index specification.
"""
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

"""
    IndexTuple{N}

A specification of `N` selected tensor indices, denoted by their position.
"""
const IndexTuple{N} = NTuple{N,Int}

"""
    Index2Tuple{N₁,N₂}

A specification of a permutation of `N₁ + N₂` indices that are partitioned into `N₁` left
and `N₂` right indices.
"""
const Index2Tuple{N₁,N₂} = Tuple{IndexTuple{N₁},IndexTuple{N₂}}

linearize(p::Index2Tuple) = (p[1]..., p[2]...)
total_length(::Index2Tuple{N₁,N₂}) where {N₁,N₂} = N₁ + N₂
