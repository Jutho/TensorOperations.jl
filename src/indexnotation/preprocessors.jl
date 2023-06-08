# replace all indices by a function of that index
function replaceindices((@nospecialize f), ex)
    if istensor(ex)
        if ex.head == :ref || ex.head == :typed_hcat
            if length(ex.args) == 1
                return ex
            elseif isa(ex.args[2], Expr) && ex.args[2].head == :parameters
                arg2 = ex.args[2]
                return Expr(ex.head, ex.args[1],
                            Expr(arg2.head, map(f, arg2.args)...),
                            (f(ex.args[i]) for i in 3:length(ex.args))...)
            else
                return Expr(ex.head, ex.args[1],
                            (f(ex.args[i]) for i in 2:length(ex.args))...)
            end
            return ex
        else #if ex.head == :typed_vcat
            arg2, arg3 = map((ex.args[2], ex.args[3])) do arg
                if isa(arg, Expr) && (arg.head == :row || arg.head == :tuple)
                    return Expr(arg.head, map(f, arg.args)...)
                else
                    return f(arg)
                end
            end
            return Expr(ex.head, ex.args[1], arg2, arg3)
        end
    elseif isa(ex, Expr)
        return Expr(ex.head, (replaceindices(f, e) for e in ex.args)...)
    else
        return ex
    end
end

# normalize indices with primes
function normalizeindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return ex
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return Symbol(normalizeindex(ex.args[1]), "′")
    else
        error("not a valid index: $ex")
    end
end
normalizeindices(ex::Expr) = replaceindices(normalizeindex, ex)

# replace all tensor objects by a function of that object
function replacetensorobjects(f, ex)
    # first try to replace ex completely:
    # this is needed if `ex` is a tensor object that appears outside an actual tensor
    # expression in a 'regular' block of code
    ex2 = f(ex, nothing, nothing)
    ex2 !== ex && return ex2
    if istensor(ex)
        obj, leftind, rightind = decomposetensor(ex)
        return Expr(ex.head, f(obj, leftind, rightind), ex.args[2:end]...)
    elseif isa(ex, Expr)
        return Expr(ex.head, (replacetensorobjects(f, e) for e in ex.args)...)
    else
        return ex
    end
end

# expandconj: conjugate individual terms or factors instead of a whole expression
function expandconj(ex)
    if isgeneraltensor(ex) || isscalarexpr(ex) || !isa(ex, Expr)
        return ex
    elseif isexpr(ex, :call) && ex.args[1] == :conj
        @assert length(ex.args) == 2
        return conjexpr(expandconj(ex.args[2]))
    else
        return Expr(ex.head, map(expandconj, ex.args)...)
    end
end

function conjexpr(ex)
    if isgeneraltensor(ex) || isscalarexpr(ex) || isa(ex, Symbol)
        return Expr(:call, :conj, ex)
    elseif isa(ex, Number)
        return conj(ex)
    elseif isexpr(ex, :call) && ex.args[1] == :conj
        return ex.args[2]
    elseif isexpr(ex, :call) && ex.args[1] ∈ (:*, :+, :-, :/, :\)
        return Expr(ex.head, ex.args[1], map(conjexpr, ex.args[2:end])...)
    elseif !isa(ex, Expr)
        return ex
    end
    return error("cannot conjugate expression: $ex")
end

# explicitscalar: wrap all tensor expressions with zero output indices in scalar call
function explicitscalar(ex)
    if isa(ex, Expr) # prewalk
        ex = Expr(ex.head, map(explicitscalar, ex.args)...)
    end
    if istensorexpr(ex) && isempty(getindices(ex))
        return Expr(:call, :tensorscalar, ex)
    else
        return ex
    end
end

function groupscalarfactors(ex)
    if isa(ex, Expr) # prewalk
        ex = Expr(ex.head, map(groupscalarfactors, ex.args)...)
    end
    if istensorexpr(ex) && ex.args[1] == :*
        args = ex.args[2:end]
        scalarpos = findall(isscalarexpr, args)
        length(scalarpos) == 0 && return ex
        tensorpos = setdiff(1:length(args), scalarpos)
        if length(scalarpos) == 1
            scalar = args[scalarpos[1]]
        else
            scalar = Expr(:call, :*, args[scalarpos]...)
        end
        return Expr(:call, :*, scalar, args[tensorpos]...)
    end
    return ex
end

# extracttensorobjects: replace tensor objects which are not simple symbols with newly 
# generated symbols, and assign them before the expression and after the expression as necessary
function extracttensorobjects(ex)
    inputtensors = filter!(obj -> !isa(obj, Symbol), getinputtensorobjects(ex))
    outputtensors = filter!(obj -> !isa(obj, Symbol), getoutputtensorobjects(ex))
    newtensors = filter!(obj -> !isa(obj, Symbol), getnewtensorobjects(ex))
    existingtensors = unique!(vcat(inputtensors, outputtensors))
    alltensors = unique!(vcat(existingtensors, newtensors))
    tensordict = Dict{Any,Any}(a => gensym(string(a)) for a in alltensors)
    pre = Expr(:block, [Expr(:(=), tensordict[a], a) for a in existingtensors]...)
    ex = replacetensorobjects((obj, leftind, rightind) -> get(tensordict, obj, obj), ex)
    post = Expr(:block,
                [Expr(:(=), a, tensordict[a])
                 for a in unique!(vcat(newtensors, outputtensors))]...)
    pre2 = Expr(:macrocall, Symbol("@notensor"),
                LineNumberNode(@__LINE__, Symbol(@__FILE__)), pre)
    post2 = Expr(:macrocall, Symbol("@notensor"),
                 LineNumberNode(@__LINE__, Symbol(@__FILE__)), post)
    return Expr(:block, pre2, ex, post2)
end

# insertcompatiblechecks: insert runtime checks for contraction
function insertcompatiblechecks(ex::Expr)
    if isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    end
    if isassignment(ex) || isdefinition(ex) || istensorexpr(ex)
        rhs = (isassignment(ex) || isdefinition(ex)) ? getrhs(ex) : ex
        #=
        at this point we can have either tensor contractions, or a sum of tensor contractions
        we should split up these sum of tensor contractions in groups which can be simply contracted
        =#

        if first(rhs.args) in (:-, :+)
            tensorgroups = rhs.args[2:end]
        else
            tensorgroups = [rhs]
        end

        lhs_indmaps = Dict{Any,Any}()
        if isassignment(ex)
            (symbol, leftinds, rightinds) = decomposegeneraltensor(getlhs(ex))
            inds = [leftinds[:]; rightinds]
            for (ii, li) in enumerate(inds)
                lhs_indmaps[li] = vcat(get(lhs_indmaps, li, []), (symbol, false, ii))
            end
        end

        for rhs in tensorgroups
            tindermap = Dict{Any,Any}()

            # if rhs is not a call, it is a tensor expression. I want to iterate over all tensors, hence this quick'n dirty line
            # essentially what I want here is gettensors; without stripping out conj()
            rhs = rhs.head == :call ? rhs.args[2:end] : [rhs]
            for symbol in rhs
                isscalarexpr(symbol) && continue

                (symbol, leftinds, rightinds, _, isc) = decomposegeneraltensor(symbol)
                inds = [leftinds[:]; rightinds]
                for (ii, li) in enumerate(inds)
                    tindermap[li] = vcat(get(tindermap, li, []), (symbol, ii, isc))
                end
            end

            for (k, v) in tindermap
                if length(v) == 1
                    lhs_indmaps[k] = vcat(get(lhs_indmaps, k, []), v)
                else
                    reference = v[1]
                    for b in v[2:end]
                        ex = quote
                            @notensor checkcontractible($(reference[1]), $(reference[2]),
                                                        $(reference[3]),
                                                        $(b[1]), $(b[2]), $(b[3]), $(k))
                            $ex
                        end
                    end
                end
            end
        end
        for (k, v) in lhs_indmaps
            reference = first(v)
            for b in v[2:end]
                ex = quote
                    @notensor checkcontractible($(reference[1]), $(!reference[2]),
                                                $(reference[3]),
                                                $(b[1]), $(b[2]), $(b[3]), $(k))
                    $ex
                end
            end
        end

        return ex
    else
        return Expr(ex.head, map(x -> insertcompatiblechecks(x), ex.args)...)
    end
end

insertcompatiblechecks(ex) = ex

const costcache = LRU{Any,Any}(; maxsize=10^5)

function costcheck(ex::Expr, source, parser, method=:warn)
    method in (:cache, :warn) || throw(ArgumentError("Invalid costcheck method."))
    if ex.head == :macrocall && ex.args[1] == Symbol("@notensor")
        return ex
    end

    if isassignment(ex) || isdefinition(ex) || istensorexpr(ex)
        rhs = (isassignment(ex) || isdefinition(ex)) ? getrhs(ex) : ex
        #=
        at this point we can have either tensor contractions, or a sum of tensor contractions
        we should split up these sum of tensor contractions in groups which can be simply contracted
        =#
        if first(rhs.args) in (:-, :+)
            tensorgroups = rhs.args[2:end]
        else
            tensorgroups = [rhs]
        end

        for (subgroup, rhs) in enumerate(tensorgroups)
            args = rhs.args[2:end]
            network = map(getindices, args)
            tree = parser.contractiontreebuilder(network) # this is the tree that would have been used
            Costexpr = Expr(:call, Expr(:curly, :Dict, :Any, :Float64))
            for (symbol, indices) in zip(args, network)
                for (ctr, ind) in enumerate(indices)
                    push!(Costexpr.args,
                          :($ind => tensorcost($(decomposegeneraltensor(symbol)[1]), $ctr)))
                end
            end

            key = "$(source.file) : $(source.line) ($subgroup)"
            cost_map = gensym()
            current_cost = gensym()
            optimal_cost = gensym()
            optimal_tree = gensym()
            optimal_order = gensym()

            if method == :cache # global costcache
                ex = quote
                    @notensor begin
                        $(cost_map) = $(Costexpr)
                        $(current_cost) = first(TensorOperations.calc_curcost($(tree),
                                                                              $(network),
                                                                              $(cost_map)))

                        if !($(key) in keys(TensorOperations.costcache)) ||
                           first(TensorOperations.costcache[$(key)]) < $(current_cost)
                            $(optimal_tree), $(optimal_cost) = TensorOperations.optimaltree($(network),
                                                                                            $(cost_map))
                            TensorOperations.costcache[$(key)] = ($(current_cost),
                                                                  $(optimal_cost),
                                                                  TensorOperations.find_index_map($(optimal_tree),
                                                                                                  $(network)))
                        end
                    end
                    $ex
                end
            elseif method == :warn
                ex = quote
                    @notensor begin
                        $cost_map = $Costexpr
                        $current_cost = first(TensorOperations.calc_curcost($tree, $network,
                                                                            $cost_map))
                        $optimal_tree, $optimal_cost = TensorOperations.optimaltree($network,
                                                                                    $cost_map)

                        if $current_cost > $optimal_cost
                            $optimal_order = TensorOperations.find_index_map($optimal_tree,
                                                                             $network)
                            @warn "Current cost: $($current_cost), Optimal cost: $($optimal_cost), Optimal order: $($optimal_order)"
                        end
                    end
                    $ex
                end
            else
                @assert false
            end
        end
        return ex
    else
        return Expr(ex.head, map(e -> costcheck(e, source, parser, method), ex.args)...)
    end
end

costcheck(ex, source, parser, method) = ex

function find_index_map(optimal_tree, network)
    pairs = collect(betterind(optimal_tree, deepcopy(network))[4])
    sort!(pairs; by=x -> x[1])
    return isempty(pairs) ? Int[] : invperm(last.(pairs))
end

function betterind(tree, indices, usedind=0)
    if isa(tree, Int)
        return tree, indices, usedind, Dict()
    else
        (lt, indices, usedind, ltm) = betterind(tree[1], indices, usedind)
        (rt, indices, usedind, rtm) = betterind(tree[2], indices, usedind)

        tocont = intersect(indices[lt], indices[rt])
        nind = Dict(zip(tocont, (usedind + 1):(usedind + length(tocont))))
        usedind += length(tocont)

        nind = merge(ltm, rtm, nind)

        curcont = length(indices) + 1
        push!(indices, symdiff(indices[lt], indices[rt]))

        return curcont, indices, usedind, nind
    end
end

function calc_curcost(tree, indices, costs)
    if isa(tree, Int)
        return 0, indices[tree]
    else
        c1, i1 = calc_curcost(tree[1], indices, costs)
        c2, i2 = calc_curcost(tree[2], indices, costs)
        cc = c1 + c2

        open = symdiff(i1, i2)
        tocontract = intersect(i1, i2)

        oc = isempty(open) ? 1 : prod([costs[i] for i in open])
        tc = isempty(tocontract) ? 0 : prod([costs[i] for i in tocontract])

        cc += oc * tc
        return cc, open
    end
end
