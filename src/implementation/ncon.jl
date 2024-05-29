"""
    ncon(tensorlist, indexlist, [conjlist, sym]; order = ..., output = ...)

Contract the tensors in `tensorlist` (of type `Vector` or `Tuple`) according to the network
as specified by `indexlist`. Here, `indexlist` is a list (i.e. a `Vector` or `Tuple`) with
the same length as `tensorlist` whose entries are themselves lists (preferably
`Vector{Int}`) where every integer entry provides a label for corresponding index/dimension
of the corresponding tensor in `tensorlist`. Positive integers are used to label indices
that need to be contracted, and such thus appear in two different entries within
`indexlist`, whereas negative integers are used to label indices of the output tensor, and
should appear only once.

Optional arguments in another list with the same length, `conjlist`, whose entries are of
type `Bool` and indicate whether the corresponding tensor object should be conjugated
(`true`) or not (`false`). The default is `false` for all entries.

By default, contractions are performed in the order such that the indices being contracted
over are labelled by increasing integers, i.e. first the contraction corresponding to label
`1` is performed. The output tensor had an index order corresponding to decreasing
(negative, so increasing in absolute value) index labels. The keyword arguments `order` and
`output` allow to change these defaults.

See also the macro version [`@ncon`](@ref).
"""
function ncon(tensors, network,
              conjlist=fill(false, length(tensors));
              order=nothing, output=nothing)
    length(tensors) == length(network) == length(conjlist) ||
        throw(ArgumentError("number of tensors and of index lists should be the same"))
    isnconstyle(network) || throw(ArgumentError("invalid NCON network: $network"))
    output′ = nconoutput(network, output)

    if length(tensors) == 1
        if length(output′) == length(network[1])
            return tensorcopy(output′, tensors[1], network[1], conjlist[1])
        else
            return tensortrace(output′, tensors[1], network[1], conjlist[1])
        end
    end

    (tensors, network) = resolve_traces(tensors, network)
    tree = order === nothing ? ncontree(network) : indexordertree(network, order)

    A, IA, CA = contracttree(tensors, network, conjlist, tree[1])
    B, IB, CB = contracttree(tensors, network, conjlist, tree[2])
    IC = tuple(output′...)

    return tensorcontract(IC, A, IA, CA, B, IB, CB)
end

function contracttree(tensors, network, conjlist, tree)
    @nospecialize
    if tree isa Int
        return tensors[tree], tuple(network[tree]...), (conjlist[tree])
    end
    A, IA, CA = contracttree(tensors, network, conjlist, tree[1])
    B, IB, CB = contracttree(tensors, network, conjlist, tree[2])
    IC = tuple(symdiff(IA, IB)...)
    C = tensorcontract(IC, A, IA, CA, B, IB, CB)
    return C, IC, false
end

function nconoutput(network, output)
    outputindices = Vector{Int}()
    for n in network
        for k in n
            if k < 0
                push!(outputindices, k)
            end
        end
    end
    isnothing(output) && return sort(outputindices; rev=true)

    issetequal(output, outputindices) ||
        throw(ArgumentError("invalid NCON network: $network -> $output"))
    return output
end
