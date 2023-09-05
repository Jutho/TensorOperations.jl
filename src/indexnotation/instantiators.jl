contract_op(args...) = +(*(args...), *(args...))
function instantiate_scalartype(ex::Expr)
    if istensor(ex)
        return Expr(:call, :scalartype, gettensorobject(ex))
    elseif isgeneraltensor(ex)
        (object, _, _, α, _) = decomposegeneraltensor(ex)
        return Expr(:call, :(Base.promote_op), :*, Expr(:call, :scalartype, object),
                    Expr(:call, :scalartype, α))
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        if length(ex.args) > 2
            return Expr(:call, :promote_add, map(instantiate_scalartype, ex.args[2:end])...)
        else
            return instantiate_scalartype(ex.args[2])
        end
    elseif isexpr(ex, :call, 3) && ex.args[1] == :* &&
           istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        return Expr(:call, :promote_contract,
                    map(instantiate_scalartype, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] ∈ (:/, :\, :*)
        return Expr(:call, :(Base.promote_op), ex.args[1],
                    map(instantiate_scalartype, ex.args[2:end])...)
    elseif ex.head == :call && ex.args[1] == :conj
        return instantiate_scalartype(ex.args[2])
    elseif isscalarexpr(ex)
        return :(scalartype($ex))
    else
        # return :(eltype($ex)) # would probably lead to doing the same operation twice
        throw(ArgumentError("unable to determine scalartype"))
    end
end
instantiate_scalartype(ex::Number) = typeof(ex)
instantiate_scalartype(ex) = Expr(:call, :scalartype, ex)

function instantiate_scalar(ex::Expr)
    if ex.head == :call && ex.args[1] == :tensorscalar
        @assert length(ex.args) == 2 && istensorexpr(ex.args[2])
        tempvar = gensym()
        returnvar = gensym()
        return quote
            $tempvar = $(instantiate(tempvar, _zero, ex.args[2], _one, [], [],
                                     TemporaryTensor))
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
function simplify_scalarmul(exa, exb)
    if exa === _one
        return exb
    elseif exb === _one
        return exa
    end
    if exa isa Number && exb isa Number
        return exa * exb
    end
    if exa isa Number && isexpr(exb, :call) && exb.args[1] == :* && exb.args[2] isa Number
        return Expr(:call, :*, exa * exb.args[2], exb.args[3:end]...)
    end
    if exb isa Number && isexpr(exa, :call) && exa.args[1] == :* && exa.args[2] isa Number
        return Expr(:call, :*, exb * exa.args[2], exa.args[3:end]...)
    end
    if exa isa Number
        return Expr(:call, :*, exa, exb)
    end
    if exb isa Number
        return Expr(:call, :*, exb, exa)
    end
    if isexpr(exa, :call) && exa.args[1] == :* && exa.args[2] isa Number &&
       isexpr(exb, :call) && exb.args[1] == :* && exb.args[2] isa Number
        return Expr(:call, :*, exa.args[2] * exb.args[2], exa.args[3:end]...,
                    exb.args[3:end]...)
    end
    return Expr(:call, :*, exa, exb)
end

@enum AllocationStrategy ExistingTensor NewTensor TemporaryTensor
function instantiate(dst, β, ex::Expr, α, leftind::Vector{Any}, rightind::Vector{Any},
                     alloc::AllocationStrategy)
    if isgeneraltensor(ex)
        return instantiate_generaltensor(dst, β, ex, α, leftind, rightind, alloc)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        if length(ex.args) == 2
            α′ = (ex.args[1] == :-) ? Expr(:call, :-, α) : α
            return instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc)
        else # linear combination
            return instantiate_linearcombination(dst, β, ex, α, leftind, rightind, alloc)
        end
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
           istensorexpr(ex.args[2]) && istensorexpr(ex.args[3])
        return instantiate_contraction(dst, β, ex, α, leftind, rightind, alloc)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
           isscalarexpr(ex.args[2]) && istensorexpr(ex.args[3])
        α′ = simplify_scalarmul(α, ex.args[2])
        return instantiate(dst, β, ex.args[3], α′, leftind, rightind, alloc)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 &&
           istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
        α′ = simplify_scalarmul(α, ex.args[3])
        return instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc)
    end
    throw(ArgumentError("problem with parsing $ex"))
end

function instantiate_generaltensor(dst, β, ex::Expr, α, leftind::Vector{Any},
                                   rightind::Vector{Any}, alloc::AllocationStrategy)
    src, srcleftind, srcrightind, α2, conj = decomposegeneraltensor(ex)
    srcind = vcat(srcleftind, srcrightind)
    conjarg = conj ? :(:C) : :(:N)

    p1 = (map(l -> findfirst(isequal(l), srcind), leftind)...,)
    p2 = (map(l -> findfirst(isequal(l), srcind), rightind)...,)
    pC = (p1, p2)

    if alloc ∈ (NewTensor, TemporaryTensor)
        TC = gensym("T_" * string(dst))
        istemporary = (alloc === TemporaryTensor)
        Tval = α === _one ? instantiate_scalartype(ex) :
               instantiate_scalartype(Expr(:call, :*, α, ex))
        out = Expr(:block, Expr(:(=), TC, Tval),
                   :($dst = tensoralloc_add($TC, $pC, $src, $conjarg, $istemporary)))
    else
        out = Expr(:block)
    end

    α′ = simplify_scalarmul(α, α2)
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
        push!(out.args,
              :($dst = tensortrace!($dst, $pC, $src, ($q1, $q2), $conjarg, $α′, $β)))
        return out
    else
        if any(isnothing, (p1..., p2...)) || !isperm((p1..., p2...)) ||
           length(srcind) != length(leftind) + length(rightind)
            err = "add: $(tuple(srcleftind..., srcrightind...)) to $(tuple(leftind..., rightind...)))"
            return :(throw(IndexError($err)))
        end
        push!(out.args, :($dst = tensoradd!($dst, $pC, $src, $conjarg, $α′, $β)))
        return out
    end
end
function instantiate_linearcombination(dst, β, ex::Expr, α, leftind::Vector{Any},
                                       rightind::Vector{Any}, alloc::AllocationStrategy)
    out = Expr(:block)
    if alloc ∈ (NewTensor, TemporaryTensor)
        TC = gensym("T_" * string(dst))
        push!(out.args, Expr(:(=), TC, instantiate_scalartype(ex)))
        α′ = (α === _one) ? Expr(:call, :one, TC) :
             Expr(:call, :*, Expr(:call, :one, TC), α)
        push!(out.args, instantiate(dst, β, ex.args[2], α′, leftind, rightind, alloc))
    else
        push!(out.args, instantiate(dst, β, ex.args[2], α, leftind, rightind, alloc))
    end
    if ex.args[1] == :- && length(ex.args) == 3
        push!(out.args,
              instantiate(dst, _one, ex.args[3], Expr(:call, :-, α), leftind, rightind,
                          ExistingTensor))
    elseif ex.args[1] == :+
        for k in 3:length(ex.args)
            push!(out.args,
                  instantiate(dst, _one, ex.args[k], α, leftind, rightind, ExistingTensor))
        end
    else
        throw(ArgumentError("unable to instantiate linear combination: $ex"))
    end
    return out
end
function instantiate_contraction(dst, β, ex::Expr, α, leftind::Vector{Any},
                                 rightind::Vector{Any}, alloc::AllocationStrategy)
    exA = ex.args[2]
    exB = ex.args[3]

    indA = getindices(exA)
    indB = getindices(exB)
    cind = intersect(indA, indB)
    indC = vcat(leftind, rightind)
    oindA = intersect(indA, indC) # in the order they appear in A
    oindB = intersect(indB, indC) # in the order they appear in B

    out = Expr(:block)
    if !isgeneraltensor(exA) || hastraceindices(exA)
        A = gensym(string(dst) * "_A")
        push!(out.args, instantiate(A, _zero, exA, _one, oindA, cind, TemporaryTensor))
        poA = ((1:length(oindA))...,)
        pcA = length(oindA) .+ ((1:length(cind))...,)
        conjA = :(:N)
        Atemp = true
    else
        A, indlA, indrA, αA, conj = decomposegeneraltensor(exA)
        indA = vcat(indlA, indrA)
        poA = (map(l -> findfirst(isequal(l), indA), oindA)...,)
        pcA = (map(l -> findfirst(isequal(l), indA), cind)...,)
        conjA = conj ? :(:C) : :(:N)
        α = simplify_scalarmul(α, αA)
        Atemp = false
    end
    if !isgeneraltensor(exB) || hastraceindices(exB)
        B = gensym(string(dst) * "_B")
        push!(out.args, instantiate(B, _zero, exB, _one, cind, oindB, TemporaryTensor))
        poB = length(cind) .+ ((1:length(oindB))...,)
        pcB = ((1:length(cind))...,)
        conjB = :(:N)
        Btemp = true
    else
        B, indlB, indrB, αB, conj = decomposegeneraltensor(exB)
        indB = vcat(indlB, indrB)
        poB = (map(l -> findfirst(isequal(l), indB), oindB)...,)
        pcB = (map(l -> findfirst(isequal(l), indB), cind)...,)
        conjB = conj ? :(:C) : :(:N)
        α = simplify_scalarmul(α, αB)
        Btemp = false
    end

    oindAB = vcat(oindA, oindB)
    p1 = (map(l -> findfirst(isequal(l), oindAB), leftind)...,)
    p2 = (map(l -> findfirst(isequal(l), oindAB), rightind)...,)

    pC = (p1, p2)
    pA = (poA, pcA)
    pB = (pcB, poB)

    if any(x -> (x === nothing), (poA..., pcA..., poB..., pcB..., p1..., p2...)) ||
       !(isperm((poA..., pcA...)) && length(indA) == length(poA) + length(pcA)) ||
       !(isperm((pcB..., poB...)) && length(indB) == length(poB) + length(pcB)) ||
       !(isperm((p1..., p2...)) && length(oindAB) == length(p1) + length(p2))
        err = "contraction: $(tuple(leftind..., rightind...)) from $(tuple(indA...,)) and $(tuple(indB...,)))"
        return :(throw(IndexError($err)))
    end
    if alloc ∈ (NewTensor, TemporaryTensor)
        TCsym = gensym("T_" * string(dst))
        TCval = Expr(:call, :promote_contract, Expr(:call, :scalartype, A),
                     Expr(:call, :scalartype, B))
        if α !== _one
            TCval = Expr(:call, :(Base.promote_op), :*, instantiate_scalartype(α), TCval)
        end
        istemporary = alloc === TemporaryTensor
        initC = Expr(:block, Expr(:(=), TCsym, TCval),
                     :($dst = tensoralloc_contract($TCsym, $pC, $A, $pA, $conjA, $B, $pB,
                                                   $conjB, $istemporary)))
        push!(out.args, initC)
    end
    push!(out.args,
          :($dst = tensorcontract!($dst, $pC, $A, $pA, $conjA, $B, $pB, $conjB, $α, $β)))
    Atemp && push!(out.args, :(tensorfree!($A)))
    Btemp && push!(out.args, :(tensorfree!($B)))
    (Atemp || Btemp) && push!(out.args, dst)
    return out
end
