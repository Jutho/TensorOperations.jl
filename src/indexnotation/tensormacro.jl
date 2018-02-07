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
            indices = Vector{Any}(length(args))
            costs = Vector{Any}(length(args))
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
    # assignment case
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhsrhs(ex)
        rhs = processcontractorder(rhs, optdata)
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
                    return deindexify!(dst, 0, rhs, 1, leftind, rightind)
                elseif ex.head == :(+=)
                    return deindexify!(dst, 1, rhs, 1, leftind, rightind)
                else
                    return deindexify!(dst, 1, rhs, -1, leftind, rightind)
                end
            else
                return Expr(:(=), dst, deindexify(rhs, leftind, rightind))
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                rhs = processcontractorder(rhs, optdata)
                return Expr(ex.head, makescalar(lhs), Expr(:call, :scalar, deindexify(rhs, [], [])))
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
        return Expr(:call, :scalar, deindexify(ex, [], []))
    end
    error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex::Symbol, optdata = nothing) = esc(ex)
tensorify(ex, optdata = nothing) = ex

function processcontractorder(ex::Expr, optdata)
    ex = Expr(ex.head, map(e->processcontractorder(e, optdata), ex.args)...)
    if ex.head == :call && ex.args[1] == :* && length(ex.args) > 3
        args = ex.args[2:end]
        network = map(getindices, args)
        if optdata == nothing
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

# deindexify!: parse tensor operations
function deindexify!(dst, β, ex::Expr, α, leftind::Vector, rightind::Vector)
    if isgeneraltensor(ex)
        src, srcleftind, srcrightind, α2, conj = makegeneraltensor(ex)
        srcind = vcat(srcleftind, srcrightind)
        conjarg = conj ? :(Val{:C}) : :(Val{:N})

        p1 = (map(l->_findfirst(equalto(l), srcind), leftind)...,)
        p2 = (map(l->_findfirst(equalto(l), srcind), rightind)...,)

        if hastraceindices(ex)
            traceind = unique(setdiff(setdiff(srcind,leftind), rightind))
            q1 = (map(l->_findfirst(equalto(l), srcind), traceind)...,)
            q2 = (map(l->_findlast(equalto(l), srcind), traceind)...,)
            if !isperm((p1...,p2...,q1...,q2...)) ||
                length(srcind) != length(leftind) + length(rightind) + 2*length(traceind)
                err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
                return :(throw(IndexError($err)))
            end
            return :(trace!($α * $α2, $src, $conjarg, $β, $dst, $p1, $p2, $q1, $q2))
        else
            if !isperm((p1...,p2...)) ||
                length(srcind) != length(leftind) + length(rightind)
                err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
                return :(throw(IndexError($err)))
            end
            return :(add!($α * $α2, $src, $conjarg, $β, $dst, $p1, $p2))
        end
    elseif ex.head == :call && ex.args[1] == :+ # addition: add one by one
        dst = deindexify!(dst, β, ex.args[2], α, leftind, rightind)
        for k = 3:length(ex.args)
            dst = deindexify!(dst, 1, ex.args[k], α, leftind, rightind)
        end
        return dst
    elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 3 # subtraction: similar
        dst = deindexify!(dst, β, ex.args[2], α, leftind, rightind)
        dst = deindexify!(dst, 1, ex.args[3], Expr(:call, :-, α), leftind, rightind)
        return dst
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # multiplication: should be pairwise by now
        @assert istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        indA = getindices(ex.args[2])
        indB = getindices(ex.args[3])
        cind = intersect(indA, indB)
        oindA = setdiff(indA, cind)
        oindB = setdiff(indB, cind)

        if isgeneraltensor(ex.args[2])
            A, indAl, indAr, αA, conj = makegeneraltensor(ex.args[2])
            conjA = conj ? :(Val{:C}) : :(Val{:N})
            indA = vcat(indAl, indAr)
            poA = (map(l->_findfirst(equalto(l), indA), oindA)...,)
            pcA = (map(l->_findfirst(equalto(l), indA), cind)...,)
        else
            A = deindexify(ex.args[2], oindA, cind)
            αA = 1
            conjA = :(Val{:N})
            indA = vcat(oindA, cind)
            poA = ((1:length(oindA))...,)
            pcA = ((length(oindA)+1:length(indA))...,)
        end
        if isgeneraltensor(ex.args[3])
            B, indBl, indBr, αB, conj = makegeneraltensor(ex.args[3])
            conjB = conj ? :(Val{:C}) : :(Val{:N})
            indB = vcat(indBl, indBr)
            poB = (map(l->_findfirst(equalto(l), indB), oindB)...,)
            pcB = (map(l->_findfirst(equalto(l), indB), cind)...,)
        else
            B = deindexify(ex.args[3], cind, oindB)
            αB = 1
            conjB = :(Val{:N})
            indB = vcat(cind, oindB)
            pcB = ((1:length(cind))...,)
            poB = ((length(cind)+1:length(indB))...,)
        end
        oindAB = vcat(oindA, oindB)
        p1 = (map(l->_findfirst(equalto(l), oindAB), leftind)...,)
        p2 = (map(l->_findfirst(equalto(l), oindAB), rightind)...,)

        if !(isperm((poA...,pcA...)) && length(indA) == length(poA)+length(pcA)) ||
            !(isperm((pcB...,poB...)) && length(indB) == length(poB)+length(pcB)) ||
            !(isperm((p1...,p2...)) && length(oindAB) == length(p1)+length(p2))
            err = "contraction: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
            return :(throw(IndexError($err)))
        end
        return :(contract!($α*$αA*$αB, $A, $conjA, $B, $conjB, $β, $dst, $poA, $pcA, $poB, $pcB, $p1, $p2))
    end
end
function deindexify(ex::Expr, leftind::Vector, rightind::Vector)
    if isgeneraltensor(ex)
        src, srcleftind, srcrightind, α, conj = makegeneraltensor(ex)
        srcind = vcat(srcleftind, srcrightind)
        conjarg = conj ? :(Val{:C}) : :(Val{:N})

        p1 = (map(l->_findfirst(equalto(l), srcind), leftind)...,)
        p2 = (map(l->_findfirst(equalto(l), srcind), rightind)...,)
        if hastraceindices(ex)
            traceind = unique(setdiff(setdiff(srcind,leftind),rightind))
            q1 = (map(l->_findfirst(equalto(l), srcind), traceind)...,)
            q2 = (map(l->_findlast(equalto(l), srcind), traceind)...,)
            if !isperm((p1...,p2...,q1...,q2...)) ||
                length(srcind) != length(leftind) + length(rightind) + 2*length(traceind)
                err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
                return :(throw(IndexError($err)))
            end
            αsym = gensym()
            dstsym = gensym()
            return :(begin
                $αsym = $α
                $dstsym = similar_from_indices(promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
                trace!($αsym, $src, $conjarg, 0, $dstsym, $p1, $p2, $q1, $q2)
            end)
        else
            if !isperm((p1...,p2...)) ||
                length(srcind) != length(leftind) + length(rightind)
                err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
                return :(throw(IndexError($err)))
            end
            αsym = gensym()
            dstsym = gensym()
            return :(begin
                $αsym = $α
                $dstsym = similar_from_indices(promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
                add!($αsym, $src, $conjarg, 0, $dstsym, $p1, $p2)
            end)
        end
    elseif ex.head == :call && ex.args[1] == :+ # addition: add one by one
        dst = deindexify(ex.args[2], leftind, rightind)
        for k = 3:length(ex.args)
            dst = deindexify!(dst, 1, ex.args[k], 1, leftind, rightind)
        end
        return dst
    elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 3 # subtraction: similar
        dst = deindexify(ex.args[2], leftind, rightind)
        dst = deindexify!(dst, 1, ex.args[3], -1, leftind, rightind)
        return dst
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # multiplication: should be pairwise by now
        @assert istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        indA = getindices(ex.args[2])
        indB = getindices(ex.args[3])
        cind = intersect(indA, indB)
        oindA = setdiff(indA, cind)
        oindB = setdiff(indB, cind)

        if isgeneraltensor(ex.args[2]) && !hastraceindices(ex.args[2])
            A, indAl, indAr, αA, conj = makegeneraltensor(ex.args[2])
            conjA = conj ? :(Val{:C}) : :(Val{:N})
            indA = vcat(indAl, indAr)
            poA = (map(l->_findfirst(equalto(l), indA), oindA)...,)
            pcA = (map(l->_findfirst(equalto(l), indA), cind)...,)
            finalizeA = false
        else
            A = deindexify(ex.args[2], oindA, cind)
            αA = 1
            conjA = :(Val{:N})
            indA = vcat(oindA, cind)
            poA = ((1:length(oindA))...,)
            pcA = ((length(oindA)+1:length(indA))...,)
            finalizeA = true
        end
        if isgeneraltensor(ex.args[3]) && !hastraceindices(ex.args[3])
            B, indBl, indBr, αB, conj = makegeneraltensor(ex.args[3])
            conjB = conj ? :(Val{:C}) : :(Val{:N})
            indB = vcat(indBl, indBr)
            poB = (map(l->_findfirst(equalto(l), indB), oindB)...,)
            pcB = (map(l->_findfirst(equalto(l), indB), cind)...,)
            finalizeB = false
        else
            B = deindexify(ex.args[3], cind, oindB)
            αB = 1
            conjB = :(Val{:N})
            indB = vcat(cind, oindB)
            pcB = ((1:length(cind))...,)
            poB = ((length(cind)+1:length(indB))...,)
            finalizeB = true
        end
        oindAB = vcat(oindA, oindB)
        p1 = (map(l->_findfirst(equalto(l), oindAB), leftind)...,)
        p2 = (map(l->_findfirst(equalto(l), oindAB), rightind)...,)

        if !(isperm((poA...,pcA...)) && length(indA) == length(poA)+length(pcA)) ||
            !(isperm((pcB...,poB...)) && length(indB) == length(poB)+length(pcB)) ||
            !(isperm((p1...,p2...)) && length(oindAB) == length(p1)+length(p2))
            err = "contract: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
            return :(throw(IndexError($err)))
        end
        Asym = gensym()
        Bsym = gensym()
        αsym = gensym()
        dstsym = gensym()
        return :(begin
            $Asym = $A
            $Bsym = $B
            $αsym = $αA * $αB
            $dstsym = similar_from_indices(promote_type(eltype($Asym), eltype($Bsym), typeof($αsym)), $poA, $poB, $p1, $p2, $Asym, $Bsym, $conjA, $conjB)
            contract!($αsym, $Asym, $conjA, $Bsym, $conjB, 0, $dstsym, $poA, $pcA, $poB, $pcB, $p1, $p2)
            # $(finalizeA ? :(finalize($Asym)) : nothing)
            # $(finalizeB ? :(finalize($Bsym)) : nothing)
            $dstsym
        end)
    end
    throw(ArgumentError("problem with parsing $ex"))
end
