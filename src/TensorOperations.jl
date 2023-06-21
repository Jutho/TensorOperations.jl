module TensorOperations

using VectorInterface
using TupleTools

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
include("implementation/functions.jl")
include("implementation/ncon.jl")
include("implementation/strided.jl")
include("implementation/indices.jl")
include("implementation/allocator.jl")

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

# Package extensions
#-------------------------
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" include("../ext/TensorOperationsChainRulesCoreExt.jl")
    end
end

end # module
