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
                    return deindexify(dst, 0, rhs, 1, leftind, rightind)
                elseif ex.head == :(+=)
                    return deindexify(dst, 1, rhs, 1, leftind, rightind)
                else
                    return deindexify(dst, 1, rhs, -1, leftind, rightind)
                end
            else
                return Expr(:(=), dst, deindexify(nothing, 0, rhs, 1, leftind, rightind, false))
            end
        elseif isassignment(ex) && isscalarexpr(lhs)
            if istensorexpr(rhs) && isempty(getindices(rhs))
                return Expr(ex.head, makescalar(lhs), Expr(:call, :scalar, deindexify(nothing, 0, rhs, 1, [], [], true)))
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
        return Expr(:call, :scalar, deindexify(ex, 1, [], []))
    end
    error("invalid syntax in @tensor macro: $ex")
end
tensorify(ex::Symbol, optdata = nothing) = esc(ex)
tensorify(ex, optdata = nothing) = ex

# expandconj: conjugate individual terms or factors instead of a whole expression
function expandconj(ex::Expr)
    if isgeneraltensor(ex)
        return ex
    end
    if ex.head == :call && ex.args[1] == :conj
        @assert length(ex.args) == 2
        ex = conjexpr(ex.args[2])
    end
    ex = Expr(ex.head, map(expandconj, ex.args)...)
end
expandconj(ex) = ex

function conjexpr(ex::Expr)
    if ex.head == :call && ex.args[1] == :conj
        return ex.args[2]
    elseif ex.head == :call
        return Expr(ex.head, ex.args[1], map(conjexpr, ex.args[2:end])...)
    else
        return Expr(:call, :conj, ex)
    end
end
conjexpr(ex::Number) = Expr(:call, :conj, ex)
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

# deindexify!: parse tensor operations
function deindexify(dst, β, ex::Expr, α, leftind::Vector, rightind::Vector, istemporary = false)
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
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3 # multiplication: should be pairwise by now
        return deindexify(dst, β, ex.args[2], Expr(:call, :/, α, makescalar(ex.args[3])), leftind, rightind, istemporary)
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3 # multiplication: should be pairwise by now
        return deindexify(dst, β, ex.args[3], Expr(:call, :\, makescalar(ex.args[2]), α), leftind, rightind, istemporary)
    end
    throw(ArgumentError("problem with parsing $ex"))
end

function deindexify_generaltensor(dst, β, ex::Expr, α, leftind::Vector, rightind::Vector, istemporary = false)
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
            $dst
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
            $dst
        end
    end
end
function deindexify_linearcombination(dst, β, ex::Expr, α, leftind::Vector, rightind::Vector, istemporary = false)
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
            ex1 = deindexify(dst, 1, ex.args[k], αnew, leftind, rightind)
            returnex = quote
                $returnex
                $ex1
            end
        end
        return quote
            $returnex
            $dst
        end
    else
        throw(ArgumentError("unable to deindexify linear combination: $ex"))
    end
end
function deindexify_contraction(dst, β, ex::Expr, α, leftind::Vector, rightind::Vector, istemporary = false)
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
    sympoA = gensym()
    sympcA = gensym()
    sympoB = gensym()
    sympcB = gensym()
    symconjA = gensym()
    symconjB = gensym()

    # prepare tensors or tensor expressions
    if dst === nothing
        TA = geteltype(exA)
        TB = geteltype(exB)
        TC = Expr(:call, :promote_type, TA, TB)
    else
        TC = Expr(:call, :eltype, dst)
    end

    if !isgeneraltensor(exA) || hastraceindices(exA)
        initA = deindexify(nothing, 0, exA, Expr(:call,:one, TC), oindA, cind, true)
        poA = ((1:length(oindA))...,)
        pcA = length(oindA) .+ ((1:length(cind))...,)
        initA = quote
            $symA = $initA
            $sympoA = $poA
            $sympcA = $pcA
            $symconjA = :N
        end
        αA = 1
    else
        A, indlA, indrA, αA, conj = makegeneraltensor(exA)
        indA = vcat(indlA, indrA)
        poA = (map(l->_findfirst(isequal(l), indA), oindA)...,)
        pcA = (map(l->_findfirst(isequal(l), indA), cind)...,)
        TA = dst === nothing ? :(float(eltype($A))) : :(eltype($dst))
        conjA = conj ? :(:C) : :(:N)
        initA = quote
            if !use_blas() || (isblascontractable($A, $poA, $pcA, $conjA) && eltype($A) == $TC)
                $symA = $A
                $sympoA = $poA
                $sympcA = $pcA
                $symconjA = $conjA
            else
                $symA = cached_similar_from_indices($(QuoteNode(symA)), $TC, $poA, $pcA, $A, $conjA)
                add!(1, $A, $conjA, 0, $symA, $poA, $pcA)
                $sympoA = $(((1:length(poA))...,))
                $sympcA = $(length(poA) .+ ((1:length(pcA))...,))
                $symconjA = :N
            end
        end
    end

    if !isgeneraltensor(exB) || hastraceindices(exB)
        initB = deindexify(nothing, 0, exB, Expr(:call,:one, TC), oindB, cind, true)
        poB = ((1:length(oindB))...,)
        pcB = length(oindB) .+ ((1:length(cind))...,)
        initB = quote
            $symB = $initB
            $sympoB = $poB
            $sympcB = $pcB
            $symconjB = :N
        end
        αB = 1
    else
        B, indlB, indrB, αB, conj = makegeneraltensor(exB)
        indB = vcat(indlB, indrB)
        poB = (map(l->_findfirst(isequal(l), indB), oindB)...,)
        pcB = (map(l->_findfirst(isequal(l), indB), cind)...,)
        conjB = conj ? :(:C) : :(:N)
        initB = quote
            if !use_blas() || (isblascontractable($B, $pcB, $poB, $conjB) && eltype($B) == $TC)
                $symB = $B
                $sympcB = $pcB
                $sympoB = $poB
                $symconjB = $conjB
            else
                $symB = cached_similar_from_indices($(QuoteNode(symB)), $TC, $pcB, $poB, $B, $conjB)
                add!(1, $B, $conjB, 0, $symB, $pcB, $poB)
                $sympcB = $(((1:length(pcB))...,))
                $sympoB = $(length(pcB) .+ ((1:length(poB))...,))
                $symconjB = :N
            end
        end
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
            initC = :($symC = cached_similar_from_indices($(QuoteNode(symC)), $TC, $sympoA, $sympoB, $p1, $p2, $symA, $symB, $symconjA, $symconjB))
        else
            initC = :($symC = similar_from_indices($TC, $sympoA, $sympoB, $p1, $p2, $symA, $symB, $symconjA, $symconjB))
        end
    else
        initC = :($symC = $dst)
    end
    symC2 = gensym()
    indC = (leftind..., rightind...)
    ip1 = (map(l->_findfirst(isequal(l), indC), oindA)...,)
    ip2 = (map(l->_findfirst(isequal(l), indC), oindB)...,)
    ip1s = ((1:length(oindA))...,)
    ip2s = length(oindA) .+ ((1:length(oindB))...,)
    contractex = quote
        if !use_blas()
            $symC2 = $symC
            contract!($α*$αA*$αB, $symA, $symconjA, $symB, $symconjB, $β, $symC2, $sympoA, $sympcA, $sympoB, $sympcB, $p1, $p2)
        elseif isblascontractable($symC, $ip1, $ip2, :D)
            $symC2 = $symC
            unsafe_contract!($α*$αA*$αB, $symA, $symconjA, $symB, $symconjB, $β, $symC2, $sympoA, $sympcA, $sympoB, $sympcB, $ip1, $ip2)
        else
            $symC2 = cached_similar_from_indices($(QuoteNode(symC2)), eltype($symC), $sympoA, $sympoB, $ip1s, $ip2s, $symA, $symB, $symconjA, $symconjB)
            unsafe_contract!(1, $symA, $symconjA, $symB, $symconjB, 0, $symC2, $sympoA, $sympcA, $sympoB, $sympcB, $ip1s, $ip2s)
            add!($α*$αA*$αB, $symC2, :N, $β, $symC, $p1, $p2)
        end
    end
    return quote
        $initA
        $initB
        $initC
        $contractex
        $symC
    end
end


#
#
# function deindexify(ex::Expr, α, leftind::Vector, rightind::Vector)
#     if isgeneraltensor(ex)
#         src, srcleftind, srcrightind, α2, conj = makegeneraltensor(ex)
#         srcind = vcat(srcleftind, srcrightind)
#         conjarg = conj ? :(Val{:C}) : :(Val{:N})
#
#         p1 = (map(l->_findfirst(isequal(l), srcind), leftind)...,)
#         p2 = (map(l->_findfirst(isequal(l), srcind), rightind)...,)
#         if hastraceindices(ex)
#             traceind = unique(setdiff(setdiff(srcind,leftind),rightind))
#             q1 = (map(l->_findfirst(isequal(l), srcind), traceind)...,)
#             q2 = (map(l->_findlast(isequal(l), srcind), traceind)...,)
#             if !isperm((p1...,p2...,q1...,q2...)) ||
#                 length(srcind) != length(leftind) + length(rightind) + 2*length(traceind)
#                 err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
#                 return :(throw(IndexError($err)))
#             end
#             αsym = gensym()
#             dstsym = gensym()
#             return :(begin
#                 $αsym = $α*$α2
#                 $dstsym = _similar_from_indices($(QuoteNode(dstsym)), promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
#                 trace!($αsym, $src, $conjarg, 0, $dstsym, $p1, $p2, $q1, $q2)
#             end)
#         else
#             if !isperm((p1...,p2...)) ||
#                 length(srcind) != length(leftind) + length(rightind)
#                 err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
#                 return :(throw(IndexError($err)))
#             end
#             αsym = gensym()
#             dstsym = gensym()
#             return :(begin
#                 $αsym = $α*$α2
#                 $dstsym = _similar_from_indices($(QuoteNode(dstsym)), promote_type(eltype($src),typeof($αsym)), $p1, $p2, $src, $conjarg)
#                 add!($αsym, $src, $conjarg, 0, $dstsym, $p1, $p2)
#             end)
#         end
#     elseif ex.head == :call && ex.args[1] == :+ # addition: add one by one
#         dst = deindexify(ex.args[2], α, leftind, rightind)
#         for k = 3:length(ex.args)
#             dst = deindexify!(dst, 1, ex.args[k], α, leftind, rightind)
#         end
#         return dst
#     elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 3 # subtraction: similar
#         dst = deindexify(ex.args[2], α, leftind, rightind)
#         dst = deindexify!(dst, 1, ex.args[3], :(-$α), leftind, rightind)
#         return dst
#     elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # multiplication: should be pairwise by now
#         if isscalarexpr(ex.args[2])
#             return deindexify(ex.args[3], Expr(:call, :*, α, ex.args[2]), leftind, rightind)
#         elseif isscalarexpr(ex.args[3])
#             return deindexify(ex.args[2], Expr(:call, :*, α, ex.args[3]), leftind, rightind)
#         end
#         @assert istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
#         indA = getindices(ex.args[2])
#         indB = getindices(ex.args[3])
#         cind = intersect(indA, indB)
#         oindA = setdiff(indA, cind)
#         oindB = setdiff(indB, cind)
#
#         if isgeneraltensor(ex.args[2]) && !hastraceindices(ex.args[2])
#             A, indAl, indAr, αA, conj = makegeneraltensor(ex.args[2])
#             conjA = conj ? :(Val{:C}) : :(Val{:N})
#             indA = vcat(indAl, indAr)
#             poA = (map(l->_findfirst(isequal(l), indA), oindA)...,)
#             pcA = (map(l->_findfirst(isequal(l), indA), cind)...,)
#             finalizeA = false
#         else
#             A = deindexify(ex.args[2], 1, oindA, cind)
#             αA = 1
#             conjA = :(Val{:N})
#             indA = vcat(oindA, cind)
#             poA = ((1:length(oindA))...,)
#             pcA = ((length(oindA)+1:length(indA))...,)
#             finalizeA = true
#         end
#         if isgeneraltensor(ex.args[3]) && !hastraceindices(ex.args[3])
#             B, indBl, indBr, αB, conj = makegeneraltensor(ex.args[3])
#             conjB = conj ? :(Val{:C}) : :(Val{:N})
#             indB = vcat(indBl, indBr)
#             poB = (map(l->_findfirst(isequal(l), indB), oindB)...,)
#             pcB = (map(l->_findfirst(isequal(l), indB), cind)...,)
#             finalizeB = false
#         else
#             B = deindexify(ex.args[3], 1, cind, oindB)
#             αB = 1
#             conjB = :(Val{:N})
#             indB = vcat(cind, oindB)
#             pcB = ((1:length(cind))...,)
#             poB = ((length(cind)+1:length(indB))...,)
#             finalizeB = true
#         end
#         oindAB = vcat(oindA, oindB)
#         p1 = (map(l->_findfirst(isequal(l), oindAB), leftind)...,)
#         p2 = (map(l->_findfirst(isequal(l), oindAB), rightind)...,)
#
#         if !(isperm((poA...,pcA...)) && length(indA) == length(poA)+length(pcA)) ||
#             !(isperm((pcB...,poB...)) && length(indB) == length(poB)+length(pcB)) ||
#             !(isperm((p1...,p2...)) && length(oindAB) == length(p1)+length(p2))
#             err = "contract: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
#             return :(throw(IndexError($err)))
#         end
#         Asym = gensym()
#         Bsym = gensym()
#         αsym = gensym()
#         dstsym = gensym()
#         return :(begin
#             $Asym = $A
#             $Bsym = $B
#             $αsym = $α * $αA * $αB
#             $dstsym = _similar_from_indices($(QuoteNode(dstsym)), promote_type(eltype($Asym), eltype($Bsym), typeof($αsym)), $poA, $poB, $p1, $p2, $Asym, $Bsym, $conjA, $conjB)
#             contract!($αsym, $Asym, $conjA, $Bsym, $conjB, 0, $dstsym, $poA, $pcA, $poB, $pcB, $p1, $p2)
#             # $(finalizeA ? :(finalize($Asym)) : nothing)
#             # $(finalizeB ? :(finalize($Bsym)) : nothing)
#             $dstsym
#         end)
#     end
#     throw(ArgumentError("problem with parsing $ex"))
# end
