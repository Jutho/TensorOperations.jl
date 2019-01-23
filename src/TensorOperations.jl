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
@nospecialize
include("indexnotation/tensormacro.jl")
include("indexnotation/tensorexpressions.jl")
include("indexnotation/ncontree.jl")
@specialize
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

# Gradients
#----------
include("gradients/backwards.jl")
using Requires
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("gradients/flux.jl")
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("gradients/zygote.jl")
end


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
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(_findfirst), typeof(identity), Int})
    precompile(Tuple{typeof(_findnext), typeof(identity), Int})
    precompile(Tuple{typeof(_ncontree!), Int, Int})
    precompile(Tuple{typeof(conjexpr), Expr})
    precompile(Tuple{typeof(deindexify_contraction), Int, Int, Expr, Int, Vector, Vector, Int})
    precompile(Tuple{typeof(deindexify_generaltensor), Int, Int, Expr, Int, Vector, Vector, Int})
    precompile(Tuple{typeof(deindexify_linearcombination), Int, Int, Expr, Int, Vector, Vector, Int})
    precompile(Tuple{typeof(deindexify), Int, Int, Expr, Int, Vector, Vector, Int})
    precompile(Tuple{typeof(deindexify), Int, Int, Expr, Int, Vector, Vector})
    precompile(Tuple{typeof(expandconj), Expr})
    precompile(Tuple{typeof(expandconj), Int})
    precompile(Tuple{typeof(getallindices), Expr})
    precompile(Tuple{typeof(getallindices), Int})
    precompile(Tuple{typeof(geteltype), Expr})
    precompile(Tuple{typeof(geteltype), Int})
    precompile(Tuple{typeof(getindices), Expr})
    precompile(Tuple{typeof(hastraceindices), Int})
    precompile(Tuple{typeof(isgeneraltensor), Expr})
    precompile(Tuple{typeof(isindex), Int})
    precompile(Tuple{typeof(isnconstyle), Vector})
    precompile(Tuple{typeof(isscalarexpr), Expr})
    precompile(Tuple{typeof(istensor), Expr})
    precompile(Tuple{typeof(istensorexpr), Int})
    precompile(Tuple{typeof(makegeneraltensor), Int})
    precompile(Tuple{typeof(makeindex), Int})
    precompile(Tuple{typeof(makescalar), Expr})
    precompile(Tuple{typeof(maketensor), Int})
    precompile(Tuple{typeof(ncontree), Vector})
    precompile(Tuple{typeof(processcontractorder), Expr, Int})
    precompile(Tuple{typeof(processcontractorder), Int, Int})
    precompile(Tuple{typeof(tensorify), Expr, Int})
    precompile(Tuple{typeof(tensorify), Expr})
    precompile(Tuple{typeof(tensorify), Int, Int})
    precompile(Tuple{typeof(tree2expr), Int, Int})
    precompile(Tuple{typeof(unique2), Array{Any, 1}})
    precompile(Tuple{typeof(unique2), Array{Int64, 1}})
    precompile(Tuple{typeof(use_blas)})
    precompile(Tuple{typeof(use_cache)})
end
_precompile_()

end # module
