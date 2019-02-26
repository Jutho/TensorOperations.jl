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
@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)
@tensoropt (a,b,c,e) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost χ for indices a,b,c,e, other indices (d,f) have cost 1
@tensoropt !(a,b,c,e) C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# cost 1 for indices a,b,c,e; other indices (d,f) have asymptotic cost χ
@tensoropt C[a,b,c,d] := A[a,e,c,f,h]*B[f,g,e,b]*C[g,d,h]
# asymptotic cost χ for all indices (a,b,c,d,e,f)
```

Note that `@tensoropt` will optimize any tensor contraction sequence it encounters
in the (block of) expressions. It will however not break apart expressions that have
been explicitly grouped with parenthesis, i.e. in
```julia
@tensoroptC[a,b,c,d] := A[a,e,c,f,h]*(B[f,g,e,b]*C[g,d,h])
```
it will always contract `B` and `C` first. For a single tensor contraction sequence,
the optimal contraction order and associated (asymptotic) cost can be obtained using
`@optimalcontractiontree`.
"""
macro tensoropt(ex::Expr)
    tensorify(ex, optdata(ex))
end
macro tensoropt(optex::Expr, ex::Expr)
    tensorify(ex, optdata(optex, ex))
end
macro optimalcontractiontree(ex::Expr)
    if isassignment(ex) || isdefinition(ex)
        _, ex = getlhsrhs(ex::Expr)
    end
    if !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
    tree, cost = optimaltree(network, optdata(ex))
    return tree, cost
end
macro optimalcontractiontree(optex::Expr, ex::Expr)
    if isassignment(ex) || isdefinition(ex)
        _, ex = getlhsrhs(ex::Expr)
    end
    if !(ex.head == :call && ex.args[1] == :*)
        error("cannot compute optimal contraction tree for this expression")
    end
    network = [getindices(ex.args[k]) for k = 2:length(ex.args)]
    tree, cost = optimaltree(network, optdata(optex, ex))
    return tree, cost
end

# Process data for @tensoropt and @optimalcontractiontree
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
                    indices[k] = makeindex(args[k].args[2])
                    costs[k] = parsecost(args[k].args[3])
                    costtype = promote_type(costtype, typeof(costs[k]))
                else
                    error("invalid index cost specification")
                end
            end
            costs = convert(Vector{costtype}, costs)
        else
            indices = map(makeindex, args)
            costtype = Power{:χ,Int}
            costs = fill(Power{:χ,Int}(1,1), length(args))
        end
        return Dict{Any, costtype}(indices[k]=>costs[k] for k = 1:length(args))
    elseif optex.head == :call && optex.args[1] == :!
        allindices = unique(getallindices(ex))
        excludeind = map(makeindex, optex.args[2:end])
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

# functions for parsing and processing tensor expressions
function tensorify(ex::Expr, optdata = nothing)
    ex = expandconj(ex)
    ex = processcontractorder(ex, optdata)
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhsrhs(ex)
        if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :throw
            return rhs
        end

        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)

            if lhs.head == :ref && length(lhs.args) == 2 && lhs.args[2] == :(:)
                if all(isa(i, Integer) && i < 0 for i in indices)
                    lhs = Expr(:ref, lhs.args[1], sort(indices, rev=true)...)
                else
                    error("cannot automatically infer index order of left hand side")
                end
            end

            if hastraceindices(lhs)
                err = "left hand side of an assignment should have unique indices: $lhs"
                return :(throw(IndexError($err)))
            end
            dst, leftind, rightind = maketensor(lhs)
            if Set(vcat(leftind,rightind)) != Set(indices)
                err = "non-matching indices between left and right hand side: $ex"
                return :(throw(IndexError($err)))
            end
            if isassignment(ex)
                if ex.head == :(=)
                    return deindexify(dst, false, rhs, true, leftind, rightind)
                elseif ex.head == :(+=)
                    return deindexify(dst, true, rhs, 1, leftind, rightind)
                else
                    return deindexify(dst, true, rhs, -1, leftind, rightind)
                end
            else
                return Expr(:(=), dst, deindexify(nothing, false, rhs, true, leftind, rightind, false))
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                return Expr(ex.head, makescalar(lhs), Expr(:call, :scalar, deindexify(nothing, false, rhs, true, [], [], true)))
            elseif isscalarexpr(rhs)
                return Expr(ex.head, makescalar(lhs), makescalar(rhs))
            end
        else
            return ex # likely an error
        end
    end

    if ex.head == :block
        return Expr(ex.head, map(x->tensorify(x, optdata), ex.args)...)
    end
    if ex.head == :for
        return Expr(ex.head, esc(ex.args[1]), tensorify(ex.args[2], optdata))
    end
    # constructions of the form: a = @tensor ...
    if isscalarexpr(ex)
        return makescalar(ex)
    end
    if istensorexpr(ex)
        if !isempty(getindices(ex))
            err = "cannot evaluate $ex to a scalar: uncontracted indices"
            return :(throw(IndexError($err)))
        end
        ex = processcontractorder(ex, optdata)
        return Expr(:call, :scalar, deindexify(nothing, false, ex, true, [], [], true))
    end
    error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex::Symbol, optdata = nothing) = esc(ex)
tensorify(ex, optdata = nothing) = ex

# expandconj: conjugate individual terms or factors instead of a whole expression
function expandconj(ex::Expr)
    if isgeneraltensor(ex) || isscalarexpr(ex)
        return ex
    elseif ex.head == :call && ex.args[1] == :conj
        @assert length(ex.args) == 2
        return conjexpr(expandconj(ex.args[2]))
    else
        return Expr(ex.head, map(expandconj, ex.args)...)
    end
end
expandconj(ex) = ex

function conjexpr(ex::Expr)
    if ex.head == :call && ex.args[1] == :conj
        return ex.args[2]
    elseif isgeneraltensor(ex) || isscalarexpr(ex)
        return Expr(:call, :conj, ex)
    elseif ex.head == :call && (ex.args[1] == :* || ex.args[1] == :+ || ex.args[1] == :-)
        return Expr(ex.head, ex.args[1], map(conjexpr, ex.args[2:end])...)
    elseif ex.head == :call && (ex.args[1] == :/ || ex.args[1] == :\)
        return Expr(ex.head, ex.args[1], map(conjexpr, ex.args[2:end])...)
    else
        error("cannot conjugate expression: $ex")
    end
end
conjexpr(ex::Number) = conj(ex)
conjexpr(ex::Symbol) = Expr(:call, :conj, ex)
conjexpr(ex) = ex

# processcontractorder: convert multi-argument multiplication into tree of pairwise multiplication
function processcontractorder(ex::Expr, optdata)
    ex = Expr(ex.head, map(e->processcontractorder(e, optdata), ex.args)...)
    if ex.head == :call && ex.args[1] == :* && length(ex.args) > 3
        args = ex.args[2:end]
        network = map(getindices, args)
        err = "invalid tensor contraction: $ex"
        for a in getallindices(ex)
            count(a in n for n in network) <= 2 || return :(throw(IndexError($err)))
        end
        if optdata === nothing
            if isnconstyle(network)
                tree = ncontree(network)
                ex = tree2expr(args, tree)
            else
                ex = Expr(:call, :*, args[1], args[2])
                for k = 3:length(args)
                    ex = Expr(:call, :*, ex, args[k])
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

# deindexify: parse tensor operations
function deindexify(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any}, istemporary = false)
    if isgeneraltensor(ex)
        return deindexify_generaltensor(dst, β, ex, α, leftind, rightind, istemporary)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-) # linear combination
        return deindexify_linearcombination(dst, β, ex, α, leftind, rightind, istemporary)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # multiplication: should be pairwise by now
        if isscalarexpr(ex.args[2])
            return deindexify(dst, β, ex.args[3], Expr(:call, :*, makescalar(ex.args[2]), α), leftind, rightind, istemporary)
        elseif isscalarexpr(ex.args[3])
            return deindexify(dst, β, ex.args[2], Expr(:call, :*, α, makescalar(ex.args[3])), leftind, rightind, istemporary)
        else
            return deindexify_contraction(dst, β, ex, α, leftind, rightind, istemporary)
        end
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        return deindexify(dst, β, ex.args[2], Expr(:call, :/, α, makescalar(ex.args[3])), leftind, rightind, istemporary)
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        return deindexify(dst, β, ex.args[3], Expr(:call, :\, makescalar(ex.args[2]), α), leftind, rightind, istemporary)
    end
    throw(ArgumentError("problem with parsing $ex"))
end

function deindexify_generaltensor(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any}, istemporary = false)
    src, srcleftind, srcrightind, α2, conj = makegeneraltensor(ex)
    srcind = vcat(srcleftind, srcrightind)
    conjarg = conj ? :(:C) : :(:N)

    p1 = (map(l->_findfirst(isequal(l), srcind), leftind)...,)
    p2 = (map(l->_findfirst(isequal(l), srcind), rightind)...,)

    αsym = gensym()
    if dst === nothing
        dst = gensym()
        if istemporary
            initex = quote
                $αsym = $α*$α2
                $dst = cached_similar_from_indices($(QuoteNode(dst)), promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
            end
        else
            initex = quote
                $αsym = $α*$α2
                $dst = similar_from_indices(promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
            end
        end
    else
        initex = :($αsym = $α*$α2)
    end

    if hastraceindices(ex)
        traceind = unique(setdiff(setdiff(srcind,leftind), rightind))
        q1 = (map(l->_findfirst(isequal(l), srcind), traceind)...,)
        q2 = (map(l->_findlast(isequal(l), srcind), traceind)...,)
        if !isperm((p1...,p2...,q1...,q2...)) ||
            length(srcind) != length(leftind) + length(rightind) + 2*length(traceind)
            err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        return quote
            $initex
            trace!($α * $α2, $src, $conjarg, $β, $dst, $p1, $p2, $q1, $q2)
            # $dst
        end
    else
        if !isperm((p1...,p2...)) ||
            length(srcind) != length(leftind) + length(rightind)
            err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        return quote
            $initex
            add!($α * $α2, $src, $conjarg, $β, $dst, $p1, $p2)
            # $dst
        end
    end
end
function deindexify_linearcombination(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any}, istemporary = false)
    if ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-) # addition: add one by one
        if dst === nothing
            αnew = Expr(:call, :*, α, Expr(:call, :one, geteltype(ex)))
            ex1 = deindexify(dst, β, ex.args[2], αnew, leftind, rightind, istemporary)
            dst = gensym()
            returnex = :($dst = $ex1)
        else
            returnex = deindexify(dst, β, ex.args[2], α, leftind, rightind, istemporary)
        end
        αnew = (ex.args[1] == :-) ? Expr(:call, :-, α) : α
        for k = 3:length(ex.args)
            ex1 = deindexify(dst, true, ex.args[k], αnew, leftind, rightind)
            returnex = quote
                $returnex
                $ex1
            end
        end
        return quote
            $returnex
            # $dst
        end
    else
        throw(ArgumentError("unable to deindexify linear combination: $ex"))
    end
end
function deindexify_contraction(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any}, istemporary = false)
    @assert ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
        istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
    exA = ex.args[2]
    exB = ex.args[3]

    indA = getindices(exA)
    indB = getindices(exB)
    cind = intersect(indA, indB)
    indC = vcat(leftind, rightind)
    oindA = intersect(indC, indA) # in the order they appear in C
    oindB = intersect(indC, indB) # in the order they appear in C

    if indC == vcat(oindB, oindA) # reorder
        exA, exB = exB, exA
        indA, indB = indB, indA
        oindA, oindB = oindB, oindA
    end

    symA = gensym()
    symB = gensym()
    symC = gensym()
    symTC = gensym()

    # prepare tensors or tensor expressions
    if dst === nothing
        TA = geteltype(exA)
        TB = geteltype(exB)
        TC = Expr(:call, :promote_type, TA, TB, :(typeof($α)))
    else
        TC = Expr(:call, :eltype, dst)
    end

    if !isgeneraltensor(exA) || hastraceindices(exA)
        initA = deindexify(nothing, false, exA, true, oindA, cind, true)
        poA = ((1:length(oindA))...,)
        pcA = length(oindA) .+ ((1:length(cind))...,)
        conjA = :(:N)
        initA = Expr(:(=), symA, initA)
        αA = 1
    else
        A, indlA, indrA, αA, conj = makegeneraltensor(exA)
        indA = vcat(indlA, indrA)
        poA = (map(l->_findfirst(isequal(l), indA), oindA)...,)
        pcA = (map(l->_findfirst(isequal(l), indA), cind)...,)
        TA = dst === nothing ? :(float(eltype($A))) : :(eltype($dst))
        conjA = conj ? :(:C) : :(:N)
        initA = Expr(:(=), symA, A)
    end

    if !isgeneraltensor(exB) || hastraceindices(exB)
        initB = deindexify(nothing, false, exB, true, oindB, cind, true)
        poB = ((1:length(oindB))...,)
        pcB = length(oindB) .+ ((1:length(cind))...,)
        conjB = :(:N)
        initB = Expr(:(=), symB, initB)
        αB = 1
    else
        B, indlB, indrB, αB, conj = makegeneraltensor(exB)
        indB = vcat(indlB, indrB)
        poB = (map(l->_findfirst(isequal(l), indB), oindB)...,)
        pcB = (map(l->_findfirst(isequal(l), indB), cind)...,)
        conjB = conj ? :(:C) : :(:N)
        initB = Expr(:(=), symB, B)
    end

    oindAB = vcat(oindA, oindB)
    p1 = (map(l->_findfirst(isequal(l), oindAB), leftind)...,)
    p2 = (map(l->_findfirst(isequal(l), oindAB), rightind)...,)
    if !(isperm((poA...,pcA...)) && length(indA) == length(poA)+length(pcA)) ||
        !(isperm((pcB...,poB...)) && length(indB) == length(poB)+length(pcB)) ||
        !(isperm((p1...,p2...)) && length(oindAB) == length(p1)+length(p2))
        err = "contraction: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
        return :(throw(IndexError($err)))
    end
    if dst === nothing
        if istemporary
            initC = :($symC = cached_similar_from_indices($(QuoteNode(symC)), $symTC, $poA, $poB, $p1, $p2, $symA, $symB, $conjA, $conjB))
        else
            initC = :($symC = similar_from_indices($symTC, $poA, $poB, $p1, $p2, $symA, $symB, $conjA, $conjB))
        end
    else
        initC = :($symC = $dst)
    end

    return quote
        $symTC = $TC
        $initA
        $initB
        $initC
        contract!($α*$αA*$αB, $symA, $conjA, $symB, $conjB, $β, $symC,
                    $poA, $pcA, $poB, $pcB, $p1, $p2,
                    $((gensym(),gensym(),gensym())))
        # $symC
    end
end
