"""
    processcontractions(ex, treebuilder, treesorter, costcheck)

Process the contractions in `ex` using the given `treebuilder` and `treesorter` functions.
This is done by first extracting a network representation from the expression, then building
and sorting the contraction trees with a given `treebuilder` and `treesorter` function, and
finally inserting the contraction trees back into the expression. When the `costcheck`
argument equals `:warn` or `:cache` (as opposed to `:nothing`), the optimal contraction
order is computed at runtime using the actual values of [`tensorcost`](@ref) and this
optimal order is compared to the contraction order that was determined at compile time. If
the compile time order deviated from the optimal order, a warning will be printed (in case
of `costcheck == :warn`) or this particular contraction will be recorded in
`TensorOperations.costcache` (in case of `costcheck == :cache`). Both the warning or the 
recorded cache entry contain a `order` suggestion that can be passed to the `@tensor` macro
in order to encode the optimal contraction order at compile time..
"""
function processcontractions(ex, treebuilder, treesorter, costcheck)
    if isexpr(ex, :block)
        args = map(x -> processcontractions(x, treebuilder, treesorter, costcheck), ex.args)
        return Expr(:block, args...)
    elseif isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar
        return Expr(:call, :tensorscalar,
                    processcontractions(ex.args[2], treebuilder, treesorter, costcheck))
    elseif isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        rhs, pre, post = _processcontractions(rhs, treebuilder, treesorter, costcheck)
        if isnothing(pre)
            return Expr(ex.head, lhs, rhs)
        else
            obj, = decomposetensor(lhs)
            return Expr(:block, pre, Expr(ex.head, lhs, rhs), post, obj)
        end
    elseif istensorexpr(ex)
        ex, pre, post = _processcontractions(ex, treebuilder, treesorter, costcheck)
        if isnothing(pre)
            return ex
        else
            lhs = gensym()
            return Expr(:block, pre, Expr(:(=), lhs, ex), post, lhs)
        end
    else
        return ex
    end
end

function _processcontractions(ex, treebuilder, treesorter, costcheck)
    networks, networkexprs = getcontractionnetworks(ex)
    trees = Vector{Any}(undef, length(networks))
    treeexprdict = Dict{Any,Any}()
    for i in 1:length(networks)
        network = networks[i]
        tree = treebuilder(network)
        args = networkexprs[i].args[2:end]
        treeex = treesorter(args, tree)
        trees[i] = tree
        treeexprdict[networkexprs[i]] = treeex
    end
    ex = insertcontractiontrees(ex, treeexprdict)
    if isnothing(costcheck)
        return ex, nothing, nothing
    else
        indexmap = _fillindexmap!(Dict{Any,Any}(), ex)
        costexargs = Any[]
        for (label, v) in indexmap
            l = (label isa Symbol) ? QuoteNode(label) : label
            obj, pos, _ = v[1]
            push!(costexargs, Expr(:call, :(=>), l, Expr(:call, :tensorcost, obj, pos)))
        end
        costmapsym = gensym("costmap")
        costex = Expr(:(=), costmapsym,
                      Expr(:call, Expr(:curly, :Dict, :Any, :Float64), costexargs...))
        costcompareexprs = Vector{Any}(undef, length(networks))
        for i in 1:length(networks)
            network = networks[i]
            tree = trees[i]
            currentcostsym = gensym("currentcost")
            optimaltreesym = gensym("optimaltree")
            optimalcostsym = gensym("optimalcost")
            optimalordersym = gensym("optimalorder")
            if costcheck == :warn
                costcompareexprs[i] = :(@notensor begin
                                            $currentcostsym = first(TensorOperations.treecost($tree,
                                                                                              $network,
                                                                                              $costmapsym))
                                            $optimaltreesym, $optimalcostsym = TensorOperations.optimaltree($network,
                                                                                                            $costmapsym)
                                            if $currentcostsym > $optimalcostsym
                                                $optimalordersym = tuple(first(TensorOperations.tree2indexorder($optimaltreesym,
                                                                                                                $network))...)
                                                @warn "Tensor network: " *
                                                      $(string(networkexprs[i])) *
                                                      ":\n" *
                                                      "Current cost: $($currentcostsym), Optimal cost: $($optimalcostsym), Optimal order: $($optimalordersym)"
                                            end
                                        end)
            elseif costcheck == :cache
                key = Expr(:quote, networkexprs[i])
                costcompareexprs[i] = :(@notensor begin
                                            $currentcostsym = first(TensorOperations.treecost($tree,
                                                                                              $network,
                                                                                              $costmapsym))
                                            if !($key in
                                                 keys(TensorOperations.costcache)) ||
                                               first(TensorOperations.costcache[$key]) <
                                               $(currentcostsym)
                                                $optimaltreesym, $optimalcostsym = TensorOperations.optimaltree($network,
                                                                                                                $costmapsym)
                                                $optimalordersym = tuple(first(TensorOperations.tree2indexorder($optimaltreesym,
                                                                                                                $network))...)
                                                TensorOperations.costcache[$key] = ($currentcostsym,
                                                                                    $optimalcostsym,
                                                                                    $optimalordersym)
                                            end
                                        end)
            end
        end
        pre = Expr(:macrocall, Symbol("@notensor"),
                   LineNumberNode(@__LINE__, Symbol(@__FILE__)), costex)
        post = Expr(:block, map(removelinenumbernode, costcompareexprs)...)
        return ex, pre, post
    end
end

# extract all the tensor contractions from an expression
# return a list of contraction networks, and the corresponding tensor contraction expressions
function getcontractionnetworks(ex)
    networks = Any[]
    networkexprs = Any[]
    if isexpr(ex, :call)
        args = ex.args
        for i in 2:length(args)
            subnetworks, subnetworkexprs = getcontractionnetworks(args[i])
            append!(networks, subnetworks)
            append!(networkexprs, subnetworkexprs)
        end
    end
    if istensorcontraction(ex) && length(ex.args) > 3
        args = ex.args[2:end]
        network = map(getindices, args)
        for a in getallindices(ex)
            count(a in n for n in network) <= 2 ||
                throw(ArgumentError("index $a appears more than twice in tensor contraction: $ex"))
        end
        push!(networks, network)
        push!(networkexprs, ex)
    end
    return networks, networkexprs
end

function insertcontractiontrees(ex, treeexprdict)
    if istensorcontraction(ex) && length(ex.args) > 3
        ex = treeexprdict[ex]
    end
    if isexpr(ex, :call)
        args = ex.args
        nargs = length(args)
        ex = Expr(:call, args[1],
                  (insertcontractiontrees(args[i], treeexprdict) for i in 2:nargs)...)
    end
    return ex
end

function treecost(tree, network, costs)
    if isa(tree, Int)
        return 0, network[tree]
    else
        c1, i1 = treecost(tree[1], network, costs)
        c2, i2 = treecost(tree[2], network, costs)
        cost = c1 + c2

        open = symdiff(i1, i2)
        tocontract = intersect(i1, i2)

        oc = isempty(open) ? 1 : prod([costs[i] for i in open])
        tc = isempty(tocontract) ? 0 : prod([costs[i] for i in tocontract])

        cost += oc * tc
        return cost, open
    end
end

function tree2indexorder(tree, network)
    if isa(tree, Int)
        return Any[], network[tree]
    else
        indexorder1, openind1 = tree2indexorder(tree[1], network)
        indexorder2, openind2 = tree2indexorder(tree[2], network)
        openind = symdiff(openind1, openind2)
        indexorder = vcat(indexorder1, indexorder2, intersect(openind1, openind2))
        return indexorder, openind
    end
end

function defaulttreesorter(args, tree)
    if isa(tree, Int)
        return args[tree]
    else
        return Expr(:call, :*,
                    defaulttreesorter(args, tree[1]),
                    defaulttreesorter(args, tree[2]))
    end
end

function defaulttreebuilder(network)
    if isnconstyle(network)
        tree = ncontree(network)
    else
        tree = Any[1, 2]
        for k in 3:length(network)
            tree = Any[tree, k]
        end
    end
    return tree
end
