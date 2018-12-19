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
const _use_blas = Ref(true)
use_blas() = _use_blas[]
function disable_blas()
    _use_blas[] = false
    return
end
function enable_blas()
    _use_blas[] = true
    return
end

# A cache for temporaries of tensor contractions
const cache = LRU{Symbol,Any}()
const _use_cache = Ref(true)
use_cache() = _use_cache[]

"""
    disable_cache()

Disable the cache for further use but does not clear its current contents.
Also see [`clear_cache()`](@ref)
"""
function disable_cache()
    _use_cache[] = false
    return
end

"""
    enable_cache(; maxsize::Int = ..., maxrelsize::Real = 0.5)

(Re)-enable the cache for further use; set the maximal size `maxsize` (as number of bytes)
or relative size `maxrelsize`, as a fraction between 0 and 1, resulting in
`maxsize = floor(Int, maxrelsize * Sys.total_memory())`.
"""
function enable_cache(; kwargs...)
    _use_cache[] = true
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
precompile(Tuple{typeof(TensorOperations.conjexpr), Expr})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Expr, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Expr, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Expr, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Expr, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Expr, Bool, Expr, Int64, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Bool, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Int64, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Nothing, Int64, Expr, Int64, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Symbol, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Symbol, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Symbol, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Symbol, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_contraction), Symbol, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Expr, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Expr, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Expr, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Expr, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Expr, Bool, Expr, Int64, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Bool, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Int64, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Nothing, Int64, Expr, Int64, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Symbol, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Symbol, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Symbol, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Symbol, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_generaltensor), Symbol, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Expr, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Expr, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Expr, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Expr, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Expr, Bool, Expr, Int64, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Bool, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Int64, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Nothing, Int64, Expr, Int64, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Symbol, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Symbol, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Symbol, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Symbol, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify_linearcombination), Symbol, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Int64, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Expr, Bool, Expr, Int64, Array{Symbol, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Bool, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Int64, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Nothing, Int64, Expr, Int64, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Bool, Array{Int64, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Bool, Array{Symbol, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Any, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Int64, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}, Bool})
precompile(Tuple{typeof(TensorOperations.deindexify), Symbol, Bool, Expr, Expr, Array{Symbol, 1}, Array{Any, 1}})
precompile(Tuple{typeof(TensorOperations.expandconj), LineNumberNode})
precompile(Tuple{typeof(TensorOperations.expandconj), Symbol})
precompile(Tuple{typeof(TensorOperations.getallindices), Expr})
precompile(Tuple{typeof(TensorOperations.getallindices), Symbol})
precompile(Tuple{typeof(TensorOperations.geteltype), Expr})
precompile(Tuple{typeof(TensorOperations.geteltype), Float64})
precompile(Tuple{typeof(TensorOperations.getindices), Expr})
precompile(Tuple{typeof(TensorOperations.getindices), Symbol})
precompile(Tuple{typeof(TensorOperations.hastraceindices), Expr})
precompile(Tuple{typeof(TensorOperations.isdefinition), Expr})
precompile(Tuple{typeof(TensorOperations.isnconstyle), Array{Array{Any, 1}, 1}})
precompile(Tuple{typeof(TensorOperations.isscalarexpr), Expr})
precompile(Tuple{typeof(TensorOperations.istensor), Expr})
precompile(Tuple{typeof(TensorOperations.makegeneraltensor), Expr})
precompile(Tuple{typeof(TensorOperations.makeindex), Expr})
precompile(Tuple{typeof(TensorOperations.makeindex), Int64})
precompile(Tuple{typeof(TensorOperations.makeindex), Symbol})
precompile(Tuple{typeof(TensorOperations.makescalar), Expr})
precompile(Tuple{typeof(TensorOperations.maketensor), Expr})
precompile(Tuple{typeof(TensorOperations.optdata), Expr})
precompile(Tuple{typeof(TensorOperations.parsecost), Expr})
precompile(Tuple{typeof(TensorOperations.processcontractorder), Expr, Nothing})
precompile(Tuple{typeof(TensorOperations.tensorify), Expr, Nothing})
precompile(Tuple{typeof(TensorOperations.tensorify), Expr})
precompile(Tuple{typeof(TensorOperations.use_blas)})
precompile(Tuple{typeof(TensorOperations.use_cache)})
#
end # module
