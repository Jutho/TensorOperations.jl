function instantiate_scalartype(ex::Expr)
    if istensor(ex)
        return Expr(:call, :scalartype, gettensorobject(ex))
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :- || ex.args[1] == :* || ex.args[1] == :/)
        if length(ex.args) > 2
            return Expr(:call, :promote_type, map(instantiate_scalartype, ex.args[2:end])...)
        else
            return instantiate_scalartype(ex.args[2])
        end
    elseif ex.head == :call && ex.args[1] == :conj
        return instantiate_scalartype(ex.args[2])
    elseif isscalarexpr(ex)
        return :(typeof($ex))
    else
        # return :(eltype($ex)) # would probably lead to doing the same operation twice
        throw(ArgumentError("unable to determine scalartype"))
    end
end
instantiate_scalartype(ex) = Expr(:call, :typeof, ex)

function instantiate_scalar(ex::Expr)
    if ex.head == :call && ex.args[1] == :tensorscalar
        @assert length(ex.args) == 2 && istensorexpr(ex.args[2])
        tempvar = gensym()
        returnvar = gensym()
        return quote
            $tempvar = $((instantiate(nothing, 0, ex.args[2], 1, [], [], true)))
            $returnvar = tensorscalar($tempvar)
            tensorfree!($tempvar)
            $returnvar
        end
    elseif ex.head == :call
        return Expr(ex.head, ex.args[1], map(instantiate_scalar, ex.args[2:end])...)
    else
        return Expr(ex.head, map(instantiate_scalar, ex.args)...)
    end
end
instantiate_scalar(ex::Symbol) = ex
instantiate_scalar(ex) = ex

function instantiate(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
                     istemporary=false)
    if isgeneraltensor(ex)
        return instantiate_generaltensor(dst, β, ex, α, leftind, rightind, istemporary)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-) # linear combination
        return instantiate_linearcombination(dst, β, ex, α, leftind, rightind, istemporary)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # multiplication: should be pairwise by now
        if isscalarexpr(ex.args[2])
            return instantiate(dst, β, ex.args[3],
                               Expr(:call, :*, instantiate_scalar(ex.args[2]), α), leftind,
                               rightind, istemporary)
        elseif isscalarexpr(ex.args[3])
            return instantiate(dst, β, ex.args[2],
                               Expr(:call, :*, α, instantiate_scalar(ex.args[3])), leftind,
                               rightind, istemporary)
        else
            return instantiate_contraction(dst, β, ex, α, leftind, rightind, istemporary)
        end
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        return instantiate(dst, β, ex.args[2],
                           Expr(:call, :/, α, instantiate_scalar(ex.args[3])), leftind,
                           rightind, istemporary)
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        return instantiate(dst, β, ex.args[3],
                           Expr(:call, :\, instantiate_scalar(ex.args[2]), α), leftind,
                           rightind, istemporary)
    end
    throw(ArgumentError("problem with parsing $ex"))
end

function instantiate_generaltensor(dst, β, ex::Expr, α, leftind::Vector{Any},
                                   rightind::Vector{Any}, istemporary=false)
    src, srcleftind, srcrightind, α2, conj = decomposegeneraltensor(ex)
    srcind = vcat(srcleftind, srcrightind)
    conjarg = conj ? :(:C) : :(:N)

    p1 = (map(l->findfirst(isequal(l), srcind), leftind)...,)
    p2 = (map(l->findfirst(isequal(l), srcind), rightind)...,)
    pC = (p1, p2)

    αsym = gensym(:α)
    if dst === nothing
        dst = gensym(:dst)
        if istemporary
            initex = quote
                $αsym = $α*$α2
                $dst = tensoralloctemp(promote_type(scalartype($src), typeof($αsym)), $pC, $src, $conjarg)
            end
        else
            initex = quote
                $αsym = $α*$α2
                $dst = tensoralloc(promote_type(scalartype($src), typeof($αsym)), $pC, $src, $conjarg)
            end
        end
    else
        initex = :($αsym = $α * $α2)
    end

    if hastraceindices(ex)
        traceind = unique(setdiff(setdiff(srcind, leftind), rightind))
        q1 = (map(l -> findfirst(isequal(l), srcind), traceind)...,)
        q2 = (map(l -> findlast(isequal(l), srcind), traceind)...,)
        if any(x -> (x === nothing), (p1..., p2..., q1..., q2...)) ||
           !isperm((p1..., p2..., q1..., q2...)) ||
           length(srcind) != length(leftind) + length(rightind) + 2 * length(traceind)
            err = "trace: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        return quote
            $initex
            tensortrace!($dst, ($p1, $p2), $src, ($q1, $q2), $conjarg, $α * $α2, $β)
            # $dst
        end
    else
        if any(x -> (x === nothing), (p1..., p2...)) || !isperm((p1..., p2...)) ||
           length(srcind) != length(leftind) + length(rightind)
            err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        return quote
            $initex
            tensoradd!($dst, $src, ($p1, $p2), $conjarg, $α * $α2, $β)
            # $dst
        end
    end
end
function instantiate_linearcombination(dst, β, ex::Expr, α, leftind::Vector{Any},
                                       rightind::Vector{Any}, istemporary=false)
    if ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-) # addition: add one by one
        if dst === nothing
            αnew = Expr(:call, :*, α, Expr(:call, :one, instantiate_scalartype(ex)))
            ex1 = instantiate(dst, β, ex.args[2], αnew, leftind, rightind, istemporary)
            dst = gensym(:dst)
            returnex = :($dst = $ex1)
        else
            returnex = instantiate(dst, β, ex.args[2], α, leftind, rightind, istemporary)
        end
        αnew = (ex.args[1] == :-) ? Expr(:call, :-, α) : α
        for k in 3:length(ex.args)
            ex1 = instantiate(dst, true, ex.args[k], αnew, leftind, rightind)
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
        throw(ArgumentError("unable to instantiate linear combination: $ex"))
    end
end
function instantiate_contraction(dst, β, ex::Expr, α, leftind::Vector{Any},
                                 rightind::Vector{Any}, istemporary=false)
    @assert ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
            istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
    exA = ex.args[2]
    exB = ex.args[3]

    indA = getindices(exA)
    indB = getindices(exB)
    cind = intersect(indA, indB)
    indC = vcat(leftind, rightind)
    oindA = intersect(indA, indC) # in the order they appear in A
    oindB = intersect(indB, indC) # in the order they appear in B

    symA = gensym(:A)
    symB = gensym(:B)
    symC = gensym(:C)
    symTC = gensym(:TC)

    # prepare tensors or tensor expressions
    if dst === nothing
        TA = instantiate_scalartype(exA)
        TB = instantiate_scalartype(exB)
        TC = Expr(:call, :promote_type, TA, TB, :(typeof($α)))
    else
        TC = Expr(:call, :scalartype, dst)
    end

    if !isgeneraltensor(exA) || hastraceindices(exA)
        initA = instantiate(nothing, false, exA, true, oindA, cind, true)
        poA = ((1:length(oindA))...,)
        pcA = length(oindA) .+ ((1:length(cind))...,)
        conjA = :(:N)
        initA = Expr(:(=), symA, initA)
        αA = 1
        Atemp = true
    else
        A, indlA, indrA, αA, conj = decomposegeneraltensor(exA)
        indA = vcat(indlA, indrA)
        poA = (map(l->findfirst(isequal(l), indA), oindA)...,)
        pcA = (map(l->findfirst(isequal(l), indA), cind)...,)
        TA = dst === nothing ? :(float(scalartype($A))) : :(scalartype($dst))
        conjA = conj ? :(:C) : :(:N)
        symA = A
        initA = Expr(:(=), symA, A)
        Atemp = false
    end

    if !isgeneraltensor(exB) || hastraceindices(exB)
        initB = instantiate(nothing, false, exB, true, cind, oindB, true)
        poB = length(cind) .+ ((1:length(oindB))...,)
        pcB = ((1:length(cind))...,)
        conjB = :(:N)
        initB = Expr(:(=), symB, initB)
        αB = 1
        Btemp = true
    else
        B, indlB, indrB, αB, conj = decomposegeneraltensor(exB)
        indB = vcat(indlB, indrB)
        poB = (map(l -> findfirst(isequal(l), indB), oindB)...,)
        pcB = (map(l -> findfirst(isequal(l), indB), cind)...,)
        conjB = conj ? :(:C) : :(:N)
        symB = B
        initB = Expr(:(=), symB, B)
        Btemp = false
    end

    oindAB = vcat(oindA, oindB)
    p1 = (map(l->findfirst(isequal(l), oindAB), leftind)...,)
    p2 = (map(l->findfirst(isequal(l), oindAB), rightind)...,)
    
    pC = (p1, p2)
    pA = (poA, pcA)
    pB = (pcB, poB)
    
    if any(x->(x===nothing), (poA..., pcA..., poB..., pcB..., p1..., p2...)) ||
        !(isperm((poA...,pcA...)) && length(indA) == length(poA)+length(pcA)) ||
        !(isperm((pcB...,poB...)) && length(indB) == length(poB)+length(pcB)) ||
        !(isperm((p1...,p2...)) && length(oindAB) == length(p1)+length(p2))
        err = "contraction: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
        return :(throw(IndexError($err)))
    end
    if dst === nothing
        if istemporary
            initC = :($symC = tensoralloctemp($symTC, $pC, $symA, $poA, $conjA, $symB, $poB, $conjB))
        else
            initC = :($symC = tensoralloc($symTC, $pC, $symA, $poA, $conjA, $symB, $poB, $conjB))
        end
    else
        initC = :($symC = $dst)
    end

    returnex = quote
        $symTC = $TC
        $initA
        $initB
        $initC
        tensorcontract!($symC, $pC, $symA, $pA, $conjA, $symB, $pB, $conjB, $α*$αA*$αB, $β)
    end
    
    if Atemp
        returnex = quote
            $returnex
            tensorfree!($symA)
        end
    end
    
    if Btemp
        returnex = quote
            $returnex
            tensorfree!($symB)
        end
    end
    
    return quote
        $returnex
        $symC
    end
end
