module TensorOperations

using TupleTools
using Strided
using Strided: AbstractStridedView, UnsafeStridedView
using LinearAlgebra
using LinearAlgebra: mul!, BLAS.BlasFloat

# export macro API
export @tensor, @tensoropt, @optimalcontractiontree

export enable_blas, disable_blas, enable_cache, disable_cache, clear_cache, cachesize

# export function based API
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

# Convenient type alias
const IndexTuple{N} = NTuple{N,Int}

# An exception type for reporting errors in the index specificatino
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

# Index notation
#----------------
include("indexnotation/tensormacro.jl")
include("indexnotation/tensorexpressions.jl")
include("indexnotation/ncontree.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/poly.jl")

# Implementations
#-----------------
include("implementation/indices.jl")
include("implementation/lrucache.jl")
include("implementation/tensorcache.jl")
include("implementation/stridedarray.jl")
include("implementation/diagonal.jl")

# Functions
#----------
include("functions/simple.jl")
include("functions/inplace.jl")

# Global package settings
#------------------------
# A switch for enabling/disabling the use of BLAS for tensor contractions
use_blas() = true
function disable_blas()
    @eval TensorOperations use_blas() = false
    return
end
function enable_blas()
    @eval TensorOperations use_blas() = true
    return
end

# A cache for temporaries of tensor contractions
const cache = LRU{Symbol,Any}()
use_cache() = true

"""
    disable_cache()

Disable the cache for further use but does not clear its current contents.
Also see [`clear_cache()`](@ref)
"""
function disable_cache()
    @eval TensorOperations use_cache() = false
    return
end

"""
    enable_cache(; maxsize::Int = ..., maxrelsize::Real = 0.5)

(Re)-enable the cache for further use; set the maximal size `maxsize` (as number of bytes)
or relative size `maxrelsize`, as a fraction between 0 and 1, resulting in
`maxsize = floor(Int, maxrelsize * Sys.total_memory())`.
"""
function enable_cache(; kwargs...)
    @eval TensorOperations use_cache() = true
    setsize!(cache; kwargs...)
    return
end

"""
    clear_cache()

Clear the current contents of the cache.
"""
function clear_cache()
    empty!(cache)
    return
end

"""
    cachesize()

Return the current memory size (in bytes) of all the objects in the cache.
"""
cachesize() = cache.currentsize

# Some precompile statements
#----------------------------
precompile(tensorify, (Expr,))
precompile(optdata,(Expr,))
precompile(optdata,(Expr,Expr))
#
end # module
