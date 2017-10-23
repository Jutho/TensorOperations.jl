# indexnotation/tensormacro.jl
#
# Defines the @tensor macro which switches to an index-notation environment.
"""
    @tensor(block)

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, this will be evaluated as pairwise contractions in left to right
order, unless the so-called NCON style is used (positive integers for contracted
indices and negative indices for open indices).
"""
macro tensor(ex::Expr)
    tensorify(ex)
end

"""
    @tensoropt(optex, block)
    @tensoropt(block)

Specify one or more tensor operations using Einstein's index notation. Indices can
be chosen to be arbitrary Julia variable names, or integers. When contracting several
tensors together, the macro will determine (at compile time) the optimal contraction
order depending on the cost associated to the individual indices. If no `optex` is
provided, all indices are assumed to have an abstract scaling `χ` which is optimized
in the asympotic limit of large `χ`.

The cost can be specified in the following ways:

```julia
@tensoropt C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]
# cost χ for all indices (a,b,c,d,e,f)
@tensoropt (a,b,c,e) C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]
# cost χ for indices a,b,c,e, other indices (d,f) have cost 1
@tensoropt !(a,b,c,e) C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]
# cost 1 for indices a,b,c,e, other indices (d,f) have cost χ
@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) C[a,b,c,d] := A[a,e,c,f]*B[f,d,e,b]
# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
```
"""
macro tensoropt(ex::Expr)
    tensorify(ex, optdata(ex))
end
macro tensoropt(optex::Expr, ex::Expr)
    tensorify(ex, optdata(optex, ex))
end
macro optimalcontractiontree(ex::Expr)
    if isassignment(ex) || isdefinition(ex)
        _,ex = getlhsrhs(ex::Expr)
    elseif !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
    tree, cost = optimaltree(network, optdata(ex))
    return tree, cost
end
macro optimalcontractiontree(optex::Expr, ex::Expr)
    if isassignment(ex) || isdefinition(ex)
        _,ex = getlhsrhs(ex::Expr)
    elseif !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
    tree, cost = optimaltree(network, optdata(optex, ex))
    return tree, cost
end

function optdata(ex::Expr)
    allindices = unique(getallindices(ex))
    cost = Power{:χ}(1,1)
    return Dict{Any, typeof(cost)}(i=>cost for i in allindices)
end

function optdata(optex::Expr, ex::Expr)
    if optex.head == :tuple
        isempty(optex.args) && return nothing
        args = optex.args
        if isa(args[1], Expr) && args[1].head == :call && args[1].args[1] == :(=>)
            indices = Vector{Any}(length(args))
            costs = Vector{Any}(length(args))
            costtype = typeof(parsecost(args[1].args[3]))
            for k = 1:length(args)
                if isa(args[k], Expr) && args[k].head == :call && args[k].args[1] == :(=>)
                    indices[k] = args[k].args[2]
                    costs[k] = parsecost(args[k].args[3])
                    costtype = promote_type(costtype, typeof(costs[k]))
                else
                    error("invalid index cost specification")
                end
            end
            costs = convert(Vector{costtype}, costs)
        else
            indices = args
            costtype = Power{:χ,Int}
            costs = fill(Power{:χ,Int}(1,1), length(args))
        end
        makeindices!(indices)
        return Dict{Any, costtype}(indices[k]=>costs[k] for k = 1:length(args))
    elseif optex.head == :call && optex.args[1] == :!
        allindices = unique(getallindices(ex))
        excludeind = makeindices!(optex.args[2:end])
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
    else
        error("invalid index cost specification: $ex")
    end
end
parsecost(ex::Number) = ex
parsecost(ex::Symbol) = Power{ex}(1,1)

isassignment(ex::Expr) = ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=)
function isdefinition(ex::Expr)
    #TODO: remove when := is removed
    # if ex.head == :(:=)
    #     warn(":= will likely be deprecated as assignment operator in Julia, use ≔ (\\coloneq + TAB) or go to http://github.com/Jutho/TensorOperations.jl to suggest ASCII alternatives", once=true, key=:warnaboutcoloneq)
    # end
    return ex.head == :(:=) || (ex.head == :call && ex.args[1] == :(≔))
end

function getlhsrhs(ex::Expr)
    if ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=) || ex.head == :(:=)
        return ex.args[1], ex.args[2]
    elseif ex.head == :call && ex.args[1] == :(≔)
        return ex.args[2], ex.args[3]
    else
        error("invalid assignment or definition $ex")
    end
end

function processcontractorder(ex::Expr, optdata)
    ex = Expr(ex.head, map(e->processcontractorder(e, optdata), ex.args))
    if ex.head == :call && ex.args[1] == :* && length(ex.args) > 3
        network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
        args = ex.args[2:end]
        if optdata == nothing
            if isnconstyle(network)
                tree = ncontree(network)
                ex = tree2expr(args, tree)
            else
                ex = Expr(:call, Any[:*, args[1], args[2]])
                for k = 3:length(args)
                    ex = Expr(:call, Any[:*, ex, args[k]])
                end
            end
        else
            tree, = optimaltree(network, optdata)
            ex = tree2expr(args, tree)
        end
    end
    return ex
end
processcontractorder(ex, optdata) = ex

function tree2expr(args, tree)
    if isa(tree, Int)
        return args[tree]
    else
        return Expr(:call, :*, tree2expr(args, tree[1]), tree2expr(args, tree[2]))
    end
end


function tensorify(ex::Expr, optdata = nothing)
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhsrhs(ex)
        rhs = processcontractorder(rhs, optdata)
        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)

            if lhs.head = :ref && length(lhs.args) == 2 && lhs.args[2] = :(:)
                if all(isa(i, Integer) && i < 0 for i in indices)
                    lhs = Expr(:ref, Any[lhs.args[1], sort(indices, rev=true)...])
                else
                    error("cannot automatically infer index order of left hand side")
                end
            end

            dst, leftind, rightind = maketensor(lhs)
            if isassignment(ex)
                if ex.head == :(=)
                    return deindexify!(dst, 0, rhs, 1 leftind, rightind)
                elseif ex.head == :(+=)
                    return deindexify!(dst, 1, rhs, 1 leftind, rightind)
                else
                    return deindexify!(dst, 1, rhs, -1 leftind, rightind)
                end
            else
                return Expr(:(=), dst, deindexify(rhs, leftind, rightind))
            end
        elseif isassignment(ex) && isscalar(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                rhs = processcontractorder(rhs, optdata)
                return Expr(ex.head, lhs, Expr(:call,[:scalar, deindexify(rhs, (), ())]))
            else
                return ex
            end
        else
            return ex # likely an error
        end
    end

    # constructions of the form: a = @tensor ...
    if istensorexpr(ex) && isempty(getindices(ex))
        ex = processcontractorder(ex, optdata)
        return Expr(:call,[:scalar, deindexify(ex, (), ())])
    end

    # @tensor begin ... end
    return Expr(ex.head, map(tensorify, ex.args, optdata)...)
end

function deindexify!(dst, β, ex::Expr, α, leftind, rightind, conj::Bool = false)
    if istensor(ex)
        src, srcleftind, srcrightind = maketensor(ex)
        srcind = vcat(srcleftind, srcrightind)

        p1 = (map(l->findfirst(equalto(l), srcind), leftind)...)
        p2 = (map(l->findfirst(equalto(l), srcind), rightind)...)

        conjarg = conj ? :(Val{:C}) : :(Val{:N})
        return :(add!($α, $A, $conjarg, $β, $C, $p1, $p2))
    elseif ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        deindexify!(dst, β, ex.args[2], α, leftind, rightind, !conj)
    elseif ex.head == :call && ex.args[1] == :*
        
    end
end
