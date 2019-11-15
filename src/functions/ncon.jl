function ncon(tensors, network,
                conjlist = fill(false, length(tensors)), sym = nothing ;
                order = nothing, output = nothing)
    length(tensors) >= 2 ||
        throw(ArgumentError("do not use `ncon` for less than two tensors"))
    length(tensors) == length(network) == length(conjlist) ||
        throw(ArgumentError("number of tensors and of index lists should be the same"))
    isnconstyle(network) || throw(ArgumentError("invalid NCON network: $network"))
    outputindices = Vector{Int}()
    for n in network
        for k in n
            if k < 0
                push!(outputindices, k)
            end
        end
    end
    if output === nothing
        output = sort(outputindices; rev = true)
    else
        for a in output
            a in outputindices ||
                throw(ArgumentError("invalid NCON network: $network -> $output"))
        end
        for a in outputindices
            a in output ||
                throw(ArgumentError("invalid NCON network: $network -> $output"))
        end
    end
    tree = order === nothing ? ncontree(network) : indexordertree(network, order)

    if sym !== nothing
        syma = Symbol(sym, "_a")
        symb = Symbol(sym, "_b")
    else
        syma = symb = nothing
    end
    A, IA, CA = contracttree(tensors, network, conjlist, tree[1], syma)
    B, IB, CB = contracttree(tensors, network, conjlist, tree[2], symb)
    IC = tuple(output...)

    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)
    T = promote_type(eltype(A), eltype(B))
    if sym !== nothing
        symc = Symbol(sym, "_c")
        C = cached_similar_from_indices(symc, T, oindA, oindB, indCinoAB, (), A, B, CA, CB)
    else
        C = similar_from_indices(T, oindA, oindB, indCinoAB, (), A, B, CA, CB)
    end
    if sym !== nothing
        symcontract = (Symbol(sym, "_a′"), Symbol(sym, "_b′"), Symbol(sym, "_c′"))
    else
        symcontract = nothing
    end
    contract!(true, A, CA, B, CB, false, C,
                oindA, cindA, oindB, cindB, indCinoAB, symcontract)
    return C
end

function contracttree(tensors, network, conjlist, tree, sym)
    @nospecialize
    if tree isa Int
        return tensors[tree], tuple(network[tree]...), (conjlist[tree] ? :C : :N)
    end

    if sym !== nothing
        syma = Symbol(sym, "_a")
        symb = Symbol(sym, "_b")
    else
        syma = nothing
        symb = nothing
    end
    A, IA, CA = contracttree(tensors, network, conjlist, tree[1], syma)
    B, IB, CB = contracttree(tensors, network, conjlist, tree[2], symb)
    IC = tuple(symdiff(IA, IB)...)
    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)
    T = promote_type(eltype(A), eltype(B))
    if sym !== nothing
        symc = Symbol(sym, "_c")
        C = cached_similar_from_indices(symc, T, oindA, oindB, indCinoAB, (), A, B, CA, CB)
    else
        C = similar_from_indices(T, oindA, oindB, indCinoAB, (), A, B, CA, CB)
    end
    if sym !== nothing
        symcontract = (Symbol(sym, "_a′"), Symbol(sym, "_b′"), Symbol(sym, "_c′"))
    else
        symcontract = nothing
    end
    contract!(true, A, CA, B, CB, false, C,
                oindA, cindA, oindB, cindB, indCinoAB, symcontract)
    return C, IC, :N
end
