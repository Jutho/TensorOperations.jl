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
    AVector = Vector{Any}
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    @assert precompile(Tuple{typeof(_findfirst), Base.Fix2{typeof(Base.isequal), Symbol}, AVector})
    @assert precompile(Tuple{typeof(_findnext), Base.Fix2{typeof(Base.isequal), Int64}, AVector, Int64})
    @assert precompile(Tuple{typeof(_intersect), Base.BitArray{1}, Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_intersect), Base.BitSet, Base.BitSet})
    @assert precompile(Tuple{typeof(_intersect), UInt128, UInt128})
    @assert precompile(Tuple{typeof(_intersect), UInt32, UInt32})
    @assert precompile(Tuple{typeof(_intersect), UInt64, UInt64})
    @assert precompile(Tuple{typeof(_isemptyset), Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_isemptyset), Base.BitSet})
    @assert precompile(Tuple{typeof(_isemptyset), UInt128})
    @assert precompile(Tuple{typeof(_isemptyset), UInt32})
    @assert precompile(Tuple{typeof(_isemptyset), UInt64})
    @assert precompile(Tuple{typeof(_ncontree!), AVector, Vector{Vector{Int64}}})
    @assert precompile(Tuple{typeof(_setdiff), Base.BitArray{1}, Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_setdiff), Base.BitSet, Base.BitSet})
    @assert precompile(Tuple{typeof(_setdiff), UInt128, UInt128})
    @assert precompile(Tuple{typeof(_setdiff), UInt32, UInt32})
    @assert precompile(Tuple{typeof(_setdiff), UInt64, UInt64})
    @assert precompile(Tuple{typeof(_union), Base.BitArray{1}, Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_union), Base.BitSet, Base.BitSet})
    @assert precompile(Tuple{typeof(_union), UInt128, UInt128})
    @assert precompile(Tuple{typeof(_union), UInt32, UInt32})
    @assert precompile(Tuple{typeof(_union), UInt64, UInt64})
    @assert precompile(Tuple{typeof(addcost), Power{:χ, Int64}, Power{:χ, Int64}})
    @assert precompile(Tuple{typeof(degree), Power{:x, Int64}})
    @assert precompile(Tuple{typeof(deindexify_contraction), Int, Int, Expr, Int, AVector, AVector, Int})
    @assert precompile(Tuple{typeof(deindexify_generaltensor), Int, Int, Expr, Int, AVector, AVector, Int})
    @assert precompile(Tuple{typeof(deindexify_linearcombination), Int, Int, Expr, Int, AVector, AVector, Int})
    @assert precompile(Tuple{typeof(deindexify), Int, Int, Expr, Int, AVector, AVector, Int})
    @assert precompile(Tuple{typeof(deindexify), Int, Int, Expr, Int, AVector, AVector})
    @assert precompile(Tuple{typeof(deindexify), Expr, Bool, Expr, Int64, AVector, AVector})
    @assert precompile(Tuple{typeof(deindexify), Nothing, Bool, Expr, Bool, AVector, AVector, Bool})
    @assert precompile(Tuple{typeof(deindexify), Symbol, Bool, Expr, Bool, AVector, AVector})
    @assert precompile(Tuple{typeof(disable_blas)})
    @assert precompile(Tuple{typeof(disable_cache)})
    @assert precompile(Tuple{typeof(enable_blas)})
    @assert precompile(Tuple{typeof(enable_cache)})
    @assert precompile(Tuple{typeof(expandconj), Expr})
    @assert precompile(Tuple{typeof(expandconj), Symbol})
    @assert precompile(Tuple{typeof(getallindices), Expr})
    @assert precompile(Tuple{typeof(getallindices), Int})
    @assert precompile(Tuple{typeof(getallindices), Symbol})
    @assert precompile(Tuple{typeof(geteltype), Expr})
    @assert precompile(Tuple{typeof(getindices), Symbol})
    @assert precompile(Tuple{typeof(getindices), Expr})
    @assert precompile(Tuple{typeof(getlhsrhs), Expr})
    @assert precompile(Tuple{typeof(hastraceindices), Expr})
    @assert precompile(Tuple{typeof(isassignment), Expr})
    @assert precompile(Tuple{typeof(isdefinition), Expr})
    @assert precompile(Tuple{typeof(isgeneraltensor), Expr})
    @assert precompile(Tuple{typeof(isindex), Symbol})
    @assert precompile(Tuple{typeof(isindex), Int})
    @assert precompile(Tuple{typeof(isnconstyle), Array{AVector, 1}})
    @assert precompile(Tuple{typeof(isscalarexpr), Expr})
    @assert precompile(Tuple{typeof(isscalarexpr), Float64})
    @assert precompile(Tuple{typeof(isscalarexpr), LineNumberNode})
    @assert precompile(Tuple{typeof(isscalarexpr), Symbol})
    @assert precompile(Tuple{typeof(istensor), Expr})
    @assert precompile(Tuple{typeof(istensorexpr), Expr})
    @assert precompile(Tuple{typeof(makegeneraltensor), Expr})
    @assert precompile(Tuple{typeof(makeindex), Int})
    @assert precompile(Tuple{typeof(makeindex), Symbol})
    @assert precompile(Tuple{typeof(makescalar), Expr})
    @assert precompile(Tuple{typeof(makescalar), Float64})
    @assert precompile(Tuple{typeof(maketensor), Expr})
    @assert precompile(Tuple{typeof(mulcost), Power{:χ, Int64}, Power{:χ, Int64}})
    @assert precompile(Tuple{typeof(ncontree), Vector{AVector}})
    @assert precompile(Tuple{typeof(optdata), Expr})
    @assert precompile(Tuple{typeof(optimaltree), Vector{AVector}, Base.Dict{Any, Power{:χ, Int64}}})
    @assert precompile(Tuple{typeof(parsecost), Expr})
    @assert precompile(Tuple{typeof(parsecost), Int64})
    @assert precompile(Tuple{typeof(parsecost), Symbol})
    @assert precompile(Tuple{typeof(processcontractorder), Expr, Nothing})
    @assert precompile(Tuple{typeof(processcontractorder), Expr, Int})
    @assert precompile(Tuple{typeof(processcontractorder), Int, Int})
    @assert precompile(Tuple{typeof(processcontractorder), Symbol, Nothing})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitArray{1}}, AVector, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitArray{1}}, Array{Int64, 1}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitArray{1}}, Base.Set{Int64}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitSet}, AVector, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitSet}, Array{Int64, 1}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{Base.BitSet}, Base.Set{Int64}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt128}, AVector, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt128}, Array{Int64, 1}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt128}, Base.Set{Int64}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt32}, AVector, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt32}, Array{Int64, 1}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt32}, Base.Set{Int64}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt64}, AVector, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt64}, Array{Int64, 1}, Int64})
    @assert precompile(Tuple{typeof(storeset), Type{UInt64}, Base.Set{Int64}, Int64})
    @assert precompile(Tuple{typeof(tensorify), Expr, Nothing})
    @assert precompile(Tuple{typeof(tensorify), Expr, Int})
    @assert precompile(Tuple{typeof(tensorify), Expr})
    @assert precompile(Tuple{typeof(tensorify), Int, Int})
    @assert precompile(Tuple{typeof(tree2expr), Int, Int})
    @assert precompile(Tuple{typeof(unique2), AVector})
    @assert precompile(Tuple{typeof(unique2), Array{Int64, 1}})
    @assert precompile(Tuple{typeof(use_blas)})
    @assert precompile(Tuple{typeof(use_cache)})
end
_precompile_()

end # module
