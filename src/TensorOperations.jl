module TensorOperations

using VectorInterface
using TupleTools: TupleTools, isperm, invperm

using LinearAlgebra
using LinearAlgebra: mul!, BlasFloat
using Strided
using StridedViews: isstrided
using PtrArrays
using LRUCache

using Base.Meta: isexpr

# Exports
#---------
# export macro API
export @tensor, @tensoropt, @tensoropt_verbose, @optimalcontractiontree, @notensor, @ncon
export @cutensor, @butensor

# export function based API
export ncon
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!, tensorscalar
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalartype
export tensoralloc, tensorfree!

export IndexTuple, Index2Tuple, linearize

# export debug functionality
export checkcontractible, tensorcost

# Interface and index types
#---------------------------
include("indices.jl")
include("backends.jl")
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
include("implementation/strided.jl")
include("implementation/blascontract.jl")
include("implementation/diagonal.jl")
include("implementation/base.jl")
include("implementation/indices.jl")
include("implementation/allocator.jl")

# Global variables
#------------------
const costcache = LRU{Any,Any}(; maxsize=10^5)

# Package extensions backwards compatibility
#--------------------------------------------

using PackageExtensionCompat
function __init__()
    @require_extensions

    @info """
        TensorOperations can optionally be instructed to precompile several functions, which can be used to reduce the time to first execution (TTFX).
        This is disabled by default as this can take a while on some machines, and is only relevant for contraction-heavy workloads.

        To enable or disable precompilation, you can use the following script:

        ```julia
        using TensorOperations, Preferences
        set_preferences!(TensorOperations, "precompile_workload" => true; force=true)
        ```

        This will create a `LocalPreferences.toml` file next to your current `Project.toml` file to store this setting in a persistent way.
        """ maxlog = 1
end

include("precompile.jl")

end # module
