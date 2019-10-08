function optdata(ex::Expr)
    allindices = getallindices(ex)
    cost = Power{:χ}(1,1)
    return Dict{Any, typeof(cost)}(i=>cost for i in allindices)
end

function optdata(optex::Expr, ex::Expr)
    if optex.head == :tuple
        isempty(optex.args) && return nothing
        args = optex.args
        if isa(args[1], Expr) && args[1].head == :call && args[1].args[1] == :(=>)
            indices = Vector{Any}(undef, length(args))
            costs = Vector{Any}(undef, length(args))
            costtype = typeof(parsecost(args[1].args[3]))
            for k = 1:length(args)
                if isa(args[k], Expr) && args[k].head == :call && args[k].args[1] == :(=>)
                    indices[k] = normalizeindex(args[k].args[2])
                    costs[k] = parsecost(args[k].args[3])
                    costtype = promote_type(costtype, typeof(costs[k]))
                else
                    error("invalid index cost specification")
                end
            end
            costs = convert(Vector{costtype}, costs)
        else
            indices = map(normalizeindex, args)
            costtype = Power{:χ,Int}
            costs = fill(Power{:χ,Int}(1,1), length(args))
        end
        return Dict{Any, costtype}(indices[k]=>costs[k] for k = 1:length(args))
    elseif optex.head == :call && optex.args[1] == :!
        allindices = unique(getallindices(ex))
        excludeind = map(normalizeindex, optex.args[2:end])
        cost = Power{:χ}(1,1)
        d = Dict{Any, typeof(cost)}(i=>cost for i in allindices)
        for i in excludeind
            d[i] = 1
        end
        return d
    else
        error("invalid index cost specification")
    end
end

# Process index cost specification for @tensoropt and friends
function parsecost(ex::Expr)
    if ex.head == :call && ex.args[1] == :*
        return *(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :+
        return +(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :-
        return -(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :^
        return ^(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :/
        return /(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :big
        return big(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :float
        return float(map(parsecost, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :Int128
        return Int128(map(parsecost, ex.args[2:end])...)
    else
        error("invalid index cost specification: $ex")
    end
end
parsecost(ex::Number) = ex
parsecost(ex::Symbol) = Power{ex}(1,1)
