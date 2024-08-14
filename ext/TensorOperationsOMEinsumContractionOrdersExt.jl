module TensorOperationsOMEinsumContractionOrdersExt

using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: TreeOptimizer
using OMEinsumContractionOrders
using OMEinsumContractionOrders: EinCode, NestedEinsum, SlicedEinsum, isleaf, optimize_kahypar_auto

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:GreedyMethod}, verbose::Bool) where{TDK, TDV}
    @debug "Using optimizer GreedyMethod from OMEinsumContractionOrders"
    ome_optimizer = GreedyMethod()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:KaHyParBipartite}, verbose::Bool) where{TDK, TDV}
    @debug "Using optimizer KaHyParBipartite from OMEinsumContractionOrders"
    return optimize_kahypar(network, optdata, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:TreeSA}, verbose::Bool) where{TDK, TDV}
    @debug "Using optimizer TreeSA from OMEinsumContractionOrders"
    ome_optimizer = TreeSA()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:SABipartite}, verbose::Bool) where{TDK, TDV}
    @debug "Using optimizer SABipartite from OMEinsumContractionOrders"
    ome_optimizer = SABipartite()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:ExactTreewidth}, verbose::Bool) where{TDK, TDV}
    @debug "Using optimizer ExactTreewidth from OMEinsumContractionOrders"
    ome_optimizer = ExactTreewidth()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function optimize(network, optdata::Dict{TDK, TDV}, ome_optimizer::CodeOptimizer, verbose::Bool) where{TDK, TDV}
    @assert TDV <: Number "The values of `optdata` dictionary must be of `<:Number`"

    # transform the network as EinCode
    code, size_dict = network2eincode(network, optdata)
    # optimize the contraction order using OMEinsumContractionOrders, which gives a NestedEinsum
    optcode = optimize_code(code, size_dict, ome_optimizer)

    # transform the optimized contraction order back to the network
    optimaltree = eincode2contractiontree(optcode)

    # calculate the complexity of the contraction
    cc = OMEinsumContractionOrders.contraction_complexity(optcode, size_dict)
    if verbose
        println("Optimal contraction tree: ", optimaltree)
        println(cc)
    end
    return optimaltree, 2.0^(cc.tc)
end

function optimize_kahypar(network, optdata::Dict{TDK, TDV}, verbose::Bool) where{TDK, TDV}
    @assert TDV <: Number "The values of `optdata` dictionary must be of `<:Number`"

    # transform the network as EinCode
    code, size_dict = network2eincode(network, optdata)
    # optimize the contraction order using OMEinsumContractionOrders, which gives a NestedEinsum
    optcode = optimize_kahypar_auto(code, size_dict)

    # transform the optimized contraction order back to the network
    optimaltree = eincode2contractiontree(optcode)

    # calculate the complexity of the contraction
    cc = OMEinsumContractionOrders.contraction_complexity(optcode, size_dict)
    if verbose
        println("Optimal contraction tree: ", optimaltree)
        println(cc)
    end
    return optimaltree, 2.0^(cc.tc)
end

function network2eincode(network, optdata)
    indices = unique(vcat(network...))
    new_indices = Dict([i => j for (j, i) in enumerate(indices)])
    new_network = [Int[new_indices[i] for i in t] for t in network]
    open_edges = Int[]
    # if a indices appear only once, it is an open index
    for i in indices
        if sum([i in t for t in network]) == 1
            push!(open_edges, new_indices[i])
        end
    end
    size_dict = Dict([new_indices[i] => optdata[i] for i in keys(optdata)])
    return EinCode(new_network, open_edges), size_dict
end

function eincode2contractiontree(eincode::NestedEinsum)
    if isleaf(eincode)
        return eincode.tensorindex
    else
        return [eincode2contractiontree(arg) for arg in eincode.args]
    end
end

# TreeSA returns a SlicedEinsum, with nslice = 0, so directly using the eins
function eincode2contractiontree(eincode::SlicedEinsum)
    return eincode2contractiontree(eincode.eins)
end

end