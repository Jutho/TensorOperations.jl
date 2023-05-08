module TensorOperations

using VectorInterface
using TupleTools

using LinearAlgebra
using LinearAlgebra: mul!, BLAS.BlasFloat
using Strided
using LRUCache

# Exports
#---------
# export macro API
export @tensor, @tensoropt, @tensoropt_verbose, @optimalcontractiontree, @notensor, @ncon
export @cutensor

export enable_blas, disable_blas

# export function based API
export ncon
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!, tensorscalar
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalartype
export tensoralloc, tensorfree!

export IndexTuple, Index2Tuple, linearize

# export debug functionality
export checkcontractible, tensorcost

# Index notation
#----------------
@nospecialize
include("indexnotation/utility.jl")
include("indexnotation/verifiers.jl")
include("indexnotation/analyzers.jl")
include("indexnotation/preprocessors.jl")
include("indexnotation/ncon.jl")
include("indexnotation/instantiators.jl")
include("indexnotation/postprocessors.jl")
include("indexnotation/parser.jl")
include("indexnotation/poly.jl")
include("indexnotation/optdata.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/tensormacros.jl")
include("indexnotation/cutensormacros.jl")
include("indexnotation/indexordertree.jl")
@specialize

# Implementations
#-----------------
include("implementation/interface.jl")
include("implementation/strided.jl")
# include("implementation/backends.jl")
include("implementation/indices.jl")
# include("implementation/tensorcache.jl")
include("implementation/allocator.jl")

# Functions
#-----------
include("functions/simple.jl")
include("functions/ncon.jl")
include("functions/inplace.jl")

# Global package settings
#-------------------------
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

# Initialization
#-----------------
# function __init__()
#     @static if !isdefined(Base, :get_extension)
#         # @require Strided = "5e0ebb24-38b0-5f93-81fe-25c709ecae67" begin
#         #     include("../ext/TensorOperationsStrided.jl")
#         # end

#         @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
#             @require cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1" begin
#                 if CUDA.functional() && cuTENSOR.has_cutensor()
#                     include("../ext/TensorOperationsCUDA.jl")
#                 end
#             end
#         end
#     end
# end

# Some precompile statements
#----------------------------
function _precompile_()
    AVector = Vector{Any}
    for N in 1:8
        @assert precompile(Tuple{typeof(isperm),NTuple{N,Int}})
    end
    @assert precompile(Tuple{typeof(_intersect),Base.BitArray{1},Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_intersect),Base.BitSet,Base.BitSet})
    @assert precompile(Tuple{typeof(_intersect),UInt128,UInt128})
    @assert precompile(Tuple{typeof(_intersect),UInt32,UInt32})
    @assert precompile(Tuple{typeof(_intersect),UInt64,UInt64})
    @assert precompile(Tuple{typeof(_isemptyset),Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_isemptyset),Base.BitSet})
    @assert precompile(Tuple{typeof(_isemptyset),UInt128})
    @assert precompile(Tuple{typeof(_isemptyset),UInt32})
    @assert precompile(Tuple{typeof(_isemptyset),UInt64})
    @assert precompile(Tuple{typeof(_ncontree!),AVector,Vector{Vector{Int64}}})
    @assert precompile(Tuple{typeof(_setdiff),Base.BitArray{1},Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_setdiff),Base.BitSet,Base.BitSet})
    @assert precompile(Tuple{typeof(_setdiff),UInt128,UInt128})
    @assert precompile(Tuple{typeof(_setdiff),UInt32,UInt32})
    @assert precompile(Tuple{typeof(_setdiff),UInt64,UInt64})
    @assert precompile(Tuple{typeof(_union),Base.BitArray{1},Base.BitArray{1}})
    @assert precompile(Tuple{typeof(_union),Base.BitSet,Base.BitSet})
    @assert precompile(Tuple{typeof(_union),UInt128,UInt128})
    @assert precompile(Tuple{typeof(_union),UInt32,UInt32})
    @assert precompile(Tuple{typeof(_union),UInt64,UInt64})
    @assert precompile(Tuple{typeof(_nconmacro),Int,Int,Int})
    @assert precompile(Tuple{typeof(addcost),Power{:χ,Int64},Power{:χ,Int64}})
    @assert precompile(Tuple{typeof(degree),Power{:x,Int64}})
    @assert precompile(Tuple{typeof(instantiate_contraction),Int,Int,Expr,Int,AVector,
                             AVector,Int})
    @assert precompile(Tuple{typeof(instantiate_generaltensor),Int,Int,Expr,Int,AVector,
                             AVector,Int})
    @assert precompile(Tuple{typeof(instantiate_linearcombination),Int,Int,Expr,Int,AVector,
                             AVector,Int})
    @assert precompile(Tuple{typeof(instantiate),Int,Int,Expr,Int,AVector,AVector,Int})
    @assert precompile(Tuple{typeof(instantiate),Int,Int,Expr,Int,AVector,AVector})
    @assert precompile(Tuple{typeof(instantiate),Expr,Bool,Expr,Int64,AVector,AVector})
    @assert precompile(Tuple{typeof(instantiate),Nothing,Bool,Expr,Bool,AVector,AVector,
                             Bool})
    @assert precompile(Tuple{typeof(instantiate),Symbol,Bool,Expr,Bool,AVector,AVector})
    @assert precompile(Tuple{typeof(instantiate_scalartype),Expr})
    @assert precompile(Tuple{typeof(instantiate_scalar),Expr})
    @assert precompile(Tuple{typeof(instantiate_scalar),Float64})
    @assert precompile(Tuple{typeof(expandconj),Expr})
    @assert precompile(Tuple{typeof(expandconj),Symbol})
    @assert precompile(Tuple{typeof(getallindices),Expr})
    @assert precompile(Tuple{typeof(getallindices),Int})
    @assert precompile(Tuple{typeof(getallindices),Symbol})
    @assert precompile(Tuple{typeof(getindices),Symbol})
    @assert precompile(Tuple{typeof(getindices),Expr})
    @assert precompile(Tuple{typeof(gettensorobject),Int})
    @assert precompile(Tuple{typeof(getlhs),Expr})
    @assert precompile(Tuple{typeof(getrhs),Expr})
    @assert precompile(Tuple{typeof(isindex),Expr})
    @assert precompile(Tuple{typeof(isindex),Symbol})
    @assert precompile(Tuple{typeof(isindex),Int})
    @assert precompile(Tuple{typeof(hastraceindices),Expr})
    @assert precompile(Tuple{typeof(isassignment),Expr})
    @assert precompile(Tuple{typeof(isdefinition),Expr})
    @assert precompile(Tuple{typeof(isgeneraltensor),Expr})
    @assert precompile(Tuple{typeof(istensor),Expr})
    @assert precompile(Tuple{typeof(istensorexpr),Expr})
    @assert precompile(Tuple{typeof(isnconstyle),Array{AVector,1}})
    @assert precompile(Tuple{typeof(isscalarexpr),Expr})
    @assert precompile(Tuple{typeof(isscalarexpr),Float64})
    @assert precompile(Tuple{typeof(isscalarexpr),LineNumberNode})
    @assert precompile(Tuple{typeof(isscalarexpr),Symbol})
    @assert precompile(Tuple{typeof(istensorexpr),Expr})
    @assert precompile(Tuple{typeof(isgeneraltensor),Expr})
    @assert precompile(Tuple{typeof(decomposetensor),Expr})
    @assert precompile(Tuple{typeof(normalizeindex),Int})
    @assert precompile(Tuple{typeof(normalizeindex),Symbol})
    @assert precompile(Tuple{typeof(normalizeindex),Expr})
    @assert precompile(Tuple{typeof(mulcost),Power{:χ,Int64},Power{:χ,Int64}})
    @assert precompile(Tuple{typeof(ncontree),Vector{AVector}})
    @assert precompile(Tuple{typeof(optdata),Expr})
    @assert precompile(Tuple{typeof(optdata),Expr,Expr})
    @assert precompile(Tuple{typeof(optimaltree),Vector{AVector},
                             Base.Dict{Any,Power{:χ,Int64}}})
    @assert precompile(Tuple{typeof(parsecost),Expr})
    @assert precompile(Tuple{typeof(parsecost),Int64})
    @assert precompile(Tuple{typeof(parsecost),Symbol})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitArray{1}},AVector,Int64})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitArray{1}},Array{Int64,1},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitArray{1}},Base.Set{Int64},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitSet},AVector,Int64})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitSet},Array{Int64,1},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{Base.BitSet},Base.Set{Int64},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt128},AVector,Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt128},Array{Int64,1},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt128},Base.Set{Int64},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt32},AVector,Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt32},Array{Int64,1},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt32},Base.Set{Int64},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt64},AVector,Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt64},Array{Int64,1},Int64})
    @assert precompile(Tuple{typeof(storeset),Type{UInt64},Base.Set{Int64},Int64})
    @assert precompile(Tuple{typeof(tensorify),Expr})
    @assert precompile(Tuple{typeof(extracttensorobjects),Any})
    @assert precompile(Tuple{typeof(_flatten),Expr})
    # @assert precompile(Tuple{typeof(processcontractions), Any, Any, Any})
    @assert precompile(Tuple{typeof(defaultparser),Expr})
    @assert precompile(Tuple{typeof(defaultparser),Any})
    @assert precompile(Tuple{typeof(unique2),AVector})
    @assert precompile(Tuple{typeof(unique2),Array{Int64,1}})
end
_precompile_()

end # module
