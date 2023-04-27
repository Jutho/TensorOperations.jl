"""
    struct IndexError{<:AbstractString} <: Exception
    
exception type for reporting errors in the index specification.
"""
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end


const IndexTuple{N} = NTuple{N,Int}
const Index2Tuple{N₁,N₂} = Tuple{IndexTuple{N₁},IndexTuple{N₂}}

linearize(p::Index2Tuple) = (p[1]..., p[2]...)
