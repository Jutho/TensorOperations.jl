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
numout(p::Index2Tuple) = length(p[1])
numin(p::Index2Tuple) = length(p[2])
numind(p::Index2Tuple) = numout(p) + numin(p)

trivialpermutation(p::IndexTuple{N}) where {N} = ntuple(identity, Val(N))
function trivialpermutation(p::Index2Tuple)
    return (trivialpermutation(p[1]), numout(p) .+ trivialpermutation(p[2]))
end

istrivialpermutation(p::IndexTuple) = p == trivialpermutation(p)
istrivialpermutation(p::Index2Tuple) = p == trivialpermutation(p)
