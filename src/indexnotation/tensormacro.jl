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
            costtype = Power{:chi,Int}
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
isdefinition(ex::Expr) = ex.head == :(:=) || (ex.head == :call && ex.args[1] == :(≝))

function getlhsrhs(ex::Expr)
    if ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=) || ex.head == :(:=)
        return ex.args[1], ex.args[2]
    elseif ex.head == :call && ex.args[1] == :(≝)
        return ex.args[2], ex.args[3]
    else
        error("invalid assignment or definition $ex")
    end
end

function tensorify(ex::Expr, optdata = nothing)
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        #TODO: remove when := is removed
        # if ex.head == :(:=)
        #     warn(":= will likely be deprecated as assignment operator in Julia, use ≝ (\\eqdef + TAB) or go to http://github.com/Jutho/TensorOperations.jl to suggest ASCII alternatives", once=true, key=:warnaboutcoloneq)
        # end
        lhs, rhs = getlhsrhs(ex)
        # process left hand side
        if isa(lhs, Expr) && lhs.head == :ref
            dst = esc(lhs.args[1])
            if length(lhs.args) == 2 && lhs.args[2] == :(:)
                indices = getindices(rhs)
                if all(isa(i, Integer) && i < 0 for i in indices)
                    indices = makeindices!(sort(indices, rev=true))
                else
                    error("cannot automatically infer index order of left hand side")
                end
            else
                indices = makeindices!(lhs.args[2:end])
            end
            src = ex.head == :(-=) ? tensorify(Expr(:call, :-, rhs), optdata) : tensorify(rhs, optdata)
            if isassignment(ex)
                value = ex.head == :(=) ? 0 : +1
                return :(deindexify!($dst, $src, Indices{$(tuple(indices...))}(), $value))
            else
                return :($dst = deindexify($src, Indices{$(tuple(indices...))}() ))
            end
        elseif isdefinition(ex)
            # if lhs is not an index expression, there is no difference between assignment and definition
            ex = Expr(:(=), lhs, rhs)
        end
    end
    # single tensor expression
    if ex.head == :ref
        indices = makeindices!(ex.args[2:end])
        t = esc(ex.args[1])
        return :(indexify($t, Indices{$(tuple(indices...))}() ))
    end
    # tensor contraction: structure contraction order
    if ex.head == :call && ex.args[1] == :* && length(ex.args) > 3
        network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
        if optdata == nothing
            if isnconstyle(network)
                tree = ncontree(network)
                ex = tree2expr(ex.args[2:end], tree)
            end
        else
            tree, = optimaltree(network, optdata)
            ex = tree2expr(ex.args[2:end], tree)
        end
    end
    # scalar
    if ex.head == :call && ex.args[1] == :scalar
        if length(ex.args) != 2
            error("scalar accepts only a single argument")
        end
        src = tensorify(ex.args[2])
        indices = :(Indices{()}())
        return :(scalar(deindexify($src, $indices)))
    end
    return Expr(ex.head, map(tensorify, ex.args)...)
end
tensorify(ex::Symbol) = esc(ex)
tensorify(ex) = ex

# for any index expression, get the list of uncontracted indices from that expression
function getindices(ex::Expr)
    if ex.head == :ref
        indices = makeindices!(ex.args[2:end])
        return unique2(indices)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        return getindices(ex.args[2]) # getindices on any of the args[2:end] should yield the same result
    elseif ex.head == :call && ex.args[1] == :*
        indices = getindices(ex.args[2])
        for k = 3:length(ex.args)
            append!(indices, getindices(ex.args[k]))
        end
        return unique2(indices)
    elseif ex.head == :call && length(ex.args) == 2
        return getindices(ex.args[2])
    else
        return Vector{Any}()
    end
end
getindices(ex) = Vector{Any}()

function getallindices(ex::Expr)
    if ex.head == :ref
        return makeindices!(ex.args[2:end])
    elseif !isempty(ex.args)
        return unique(mapreduce(getallindices, vcat, ex.args))
    else
        return Vector{Any}()
    end
end
getallindices(ex) = Vector{Any}()

# make the arguments of a :ref expression into a proper list of indices of type Int, Char or Symbol
function makeindices!(list::Vector)
    for i = 1:length(list)
        if isa(list[i], Expr)
            list[i] = makesymbol(list[i])
        end
        isa(list[i], Int) || isa(list[i], Symbol) || isa(list[i], Char) || error("cannot make index from $(list[i])")
    end
    return list
end
# make a symbol from an index that is itself an expression: currently only supports priming
const prime = Symbol("'")
function makesymbol(ex::Expr)
    if ex.head == prime && length(ex.args) == 1
        if isa(ex.args[1], Symbol) || isa(ex.args[1], Int)
            return Symbol(ex.args[1], "′")
        elseif isa(ex.args[1], Expr)
            return Symbol(makesymbol(ex.args[1]), "′")
        end
    # could be extended with other functionality
    end
    error("cannot make index from $ex")
end

function tree2expr(args, tree)
    if isa(tree, Int)
        return args[tree]
    else
        return Expr(:call, :*, tree2expr(args, tree[1]), tree2expr(args, tree[2]))
    end
end
