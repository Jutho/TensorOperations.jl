module TensorOperations

using TupleTools
using Strided
using Strided: AbstractStridedView, UnsafeStridedView
using LRUCache
using LinearAlgebra
using LinearAlgebra: mul!, BLAS.BlasFloat

# export macro API
export @tensor, @tensoropt, @optimalcontractiontree

export enable_blas, disable_blas, enable_cache, disable_cache

# export function based API
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

# Convenient type alias
const IndexTuple{N} = NTuple{N,Int}

# An exception type for reporting errors in the index specificatino
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

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
const __defaultcachelength__ = 50
const cache = LRU{Symbol,Any}(__defaultcachelength__)
use_cache() = true

function disable_cache()
    resize!(cache, 0)
    @eval TensorOperations use_cache() = false
    return
end
function enable_cache(;length = __defaultcachelength__)
    resize!(cache, length)
    @eval TensorOperations use_cache() = true
    return
end

# Index notation
#----------------
include("indexnotation/tensorcache.jl")
include("indexnotation/tensormacro.jl")
include("indexnotation/tensorexpressions.jl")
include("indexnotation/ncontree.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/poly.jl")

# Implementations
#-----------------
include("implementation/indices.jl")
include("implementation/stridedarray.jl")

# Functions
#----------
include("functions/simple.jl")
include("functions/inplace.jl")

precompile(tensorify, (Expr,))
precompile(optdata,(Expr,))
precompile(optdata,(Expr,Expr))
#
end # module
