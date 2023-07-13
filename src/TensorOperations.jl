module TensorOperations

using VectorInterface
using TupleTools: TupleTools, isperm, invperm

using LinearAlgebra
using LinearAlgebra: mul!, BLAS.BlasFloat
using Strided
using LRUCache

using Base.Meta: isexpr

# Exports
#---------
# export macro API
export @tensor, @tensoropt, @tensoropt_verbose, @optimalcontractiontree, @notensor, @ncon
export @cutensor

# export function based API
export ncon
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!, tensorscalar
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalartype
export tensoralloc, tensorfree!

export IndexTuple, Index2Tuple, linearize

# export debug functionality
export checkcontractible, tensorcost

# Backends for tensor operations
#--------------------------------
struct Backend{B} end # generic empty parametric struct for dispatching on different backends

# Interface and index types
#---------------------------
include("indices.jl")
include("interface.jl")

# Index notation via macros
#---------------------------
@nospecialize
include("indexnotation/verifiers.jl")
include("indexnotation/analyzers.jl")
include("indexnotation/preprocessors.jl")
include("indexnotation/postprocessors.jl")
include("indexnotation/contractiontrees.jl")
include("indexnotation/ncontree.jl")
include("indexnotation/indexordertree.jl")
include("indexnotation/poly.jl")
include("indexnotation/optdata.jl")
include("indexnotation/optimaltree.jl")
include("indexnotation/instantiators.jl")
include("indexnotation/parser.jl")
include("indexnotation/tensormacros.jl")
@specialize

# Implementations
#-----------------
include("implementation/functions.jl")
include("implementation/ncon.jl")
include("implementation/abstractarray.jl")
include("implementation/diagonal.jl")
include("implementation/strided.jl")
include("implementation/indices.jl")
include("implementation/allocator.jl")

# Global variables
#------------------
const costcache = LRU{Any,Any}(; maxsize=10^5)

# Package extensions
#-------------------------
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include("../ext/TensorOperationsChainRulesCoreExt.jl")
        @require cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1" include("../ext/TensorOperationscuTENSORExt.jl")
    end
end

end # module
