module TensorOperationsOMEinsumContractionOrdersExt

using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: TreeOptimizer
using OMEinsumContractionOrders
using OMEinsumContractionOrders: EinCode, NestedEinsum, SlicedEinsum, isleaf

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:GreedyMethod}, verbose::Bool) where{TDK, TDV}
    ome_optimizer = GreedyMethod()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:KaHyParBipartite}, verbose::Bool) where{TDK, TDV}
    # sc_target and max_group_size are simply set as 10 for now
    ome_optimizer = KaHyParBipartite(; sc_target=10, max_group_size=10)
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:TreeSA}, verbose::Bool) where{TDK, TDV}
    ome_optimizer = TreeSA()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function TO.optimaltree(network, optdata::Dict{TDK, TDV}, ::TreeOptimizer{:SABipartite}, verbose::Bool) where{TDK, TDV}
    ome_optimizer = SABipartite()
    return optimize(network, optdata, ome_optimizer, verbose)
end

function optimize(network, optdata::Dict{TDK, TDV}, ome_optimizer::CodeOptimizer, verbose::Bool) where{TDK, TDV}

    try
        TDV <: Number
    catch
        error("Dimensions of the tensors must be numbers")
    end

    # transform the network as EinCode
    code = network2eincode(network)
    # optimize the contraction order using OMEinsumContractionOrders, which gives a NestedEinsum
    optcode = optimize_code(code, optdata, ome_optimizer)

    if verbose
        cc = OMEinsumContractionOrders.contraction_complexity(optcode, optdata)
        println("Optimized contraction order: ", optcode)
        println("Contraction complexity: ", cc)
    end

    # transform the optimized contraction order back to the network
    optimaltree = eincode2contractiontree(optcode)

    return (optimaltree,)
end

function network2eincode(network)
    indices = unique(vcat(network...))
    open_edges = empty(network[1])
    # if a indices appear only once, it is an open index
    for i in indices
        if sum([i in t for t in network]) == 1
            push!(open_edges, i)
        end
    end
    return EinCode(network, open_edges)
end

function eincode2contractiontree(eincode::NestedEinsum)
    if isleaf(eincode)
        return eincode.tensorindex
    else
        return [eincode2contractiontree(arg) for arg in eincode.args]
    end
end

function eincode2contractiontree(eincode::SlicedEinsum)
    return eincode2contractiontree(eincode.eins)
end

end