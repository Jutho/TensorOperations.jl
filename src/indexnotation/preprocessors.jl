# replace all indices by a function of that index
function replaceindices((@nospecialize f), ex)
    if isexpr(ex, :block)
        return Expr(:block, map(x -> replaceindices(f, x), ex.args)...)
    elseif isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    elseif istensor(ex)
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
    elseif isexpr(ex, prime) && length(ex.args) == 1
        return Symbol(normalizeindex(ex.args[1]), "′")
    else
        error("not a valid index: $ex")
    end
end

"""
    normalizeindices(ex::Expr)

Normalize indices of an expression by replacing all indices with a prime expression `i'` by indices with a unicode prime 'i′'.
"""
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

"""
    expandconj(ex)

Expand all `conj` calls in an expression to conjugate the individual terms and factors.
"""
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

"""
    groupscalarfactors(ex)

Group all scalar factors of a tensor expression into a single scalar factor at the start of the expression.
"""
function groupscalarfactors(ex)
    # if istensor(ex) || (isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor"))
    #     return ex
    # elseif istensorexpr(ex) && ex.args[1] == :*
    #     args = ex.args[2:end]
    #     scalarpos = findall(isscalarexpr, args)
    #     tensorpos = setdiff(1:length(args), scalarpos)
    #     length(scalarpos) == 0 && return Expr(:call, :*, map(groupscalarfactors, args)...)
    #     if length(scalarpos) == 1
    #         scalar = args[scalarpos[1]]
    #     else
    #         scalar = Expr(:call, :*, args[scalarpos]...)
    #     end
    #     if length(tensorpos) == 0
    #         return scalar
    #     else
    #         return Expr(:call, :*, scalar, map(groupscalarfactors, args[tensorpos])...)
    #     end
    # elseif istensorexpr(ex) && ex.args[1] == :/
    #     xarg = Expr(:call, :/, 1, ex.args[3])
    #     if istensorexpr(ex.args[2]) && ex.args[2].args[1] == :*
    #         args = ex.args[2].args[2:end]
    #         return groupscalarfactors(Expr(:call, :*, args..., xarg))
    #     else
    #         return Expr(:call, :*, groupscalarfactors(ex.args[2]), xarg)
    #     end
    # elseif istensorexpr(ex) && ex.args[1] == :\
    #     xarg = Expr(:call, :/, 1, ex.args[2])
    #     if istensorexpr(ex.args[3]) && ex.args[2].args[1] == :*
    #         args = ex.args[3].args[2:end]
    #         return groupscalarfactors(Expr(:call, :*, xarg, args...))
    #     else
    #         return Expr(:call, :*, xarg, groupscalarfactors(ex.args[2]))
    #     end
    # elseif isa(ex, Expr)
    #     return Expr(ex.head, map(groupscalarfactors, ex.args)...)
    # end
    return ex
end

# extracttensorobjects: replace tensor objects which are not simple symbols with newly 
# generated symbols, and assign them before the expression and after the expression as necessary
"""
    extracttensorobjects(ex)
    
Extract all tensor objects which are not simple symbols with newly generated symbols, and
assign them before the expression and after the expression as necessary, in order to avoid
multiple evaluations of the expression constituting the tensor object.
"""
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

# insertcontractionchecks: insert runtime checks for contraction
"""
    insertcontractionchecks(ex)

Insert runtime checks before each contraction, which provide clearer debug information.
"""
function insertcontractionchecks(ex)
    if isexpr(ex, :block)
        return Expr(:block, map(insertcontractionchecks, ex.args)...)
    elseif isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    elseif isassignment(ex) || isdefinition(ex) || istensorexpr(ex) || isscalarexpr(ex)
        rhs = (isassignment(ex) || isdefinition(ex)) ? getrhs(ex) : ex
        indexmap = Dict{Any,Any}()
        if isassignment(ex)
            (object, indl, indr) = decomposegeneraltensor(getlhs(ex))
            inds = vcat(indl, indr)
            for (pos, label) in enumerate(inds)
                push!(get!(indexmap, label, Vector{Any}()), (object, pos, :C)) # treat lhs as conjugated tensor
            end
        end
        _fillindexmap!(indexmap, rhs)
        out = Expr(:block)
        for (label, v) in indexmap
            l = (label isa Symbol) ? QuoteNode(label) : label
            obj1, pos1, conj1 = v[1]
            for k in 2:length(v)
                obj2, pos2, conj2 = v[k]
                push!(out.args,
                      :(@notensor checkcontractible($obj1, $pos1, $(QuoteNode(conj1)),
                                                    $obj2, $pos2,
                                                    $(QuoteNode(conj2)), $l)))
            end
        end
        return Expr(:block, out, ex)
    end
    return ex
end
function _fillindexmap!(indexmap, ex)
    if isgeneraltensor(ex)
        (object, indl, indr, _, conj) = decomposegeneraltensor(ex)
        inds = vcat(indl, indr)
        for (pos, label) in enumerate(inds)
            push!(get!(indexmap, label, Vector{Any}()), (object, pos, conj ? :C : :N))
        end
    elseif ex isa Expr
        for exa in ex.args
            _fillindexmap!(indexmap, exa)
        end
    end
    return indexmap
end
