# parsing indices in an @tensor expression:
const prime = Symbol("'")

# test for a valid index
function isindex(ex)
    if isa(ex, Symbol) || isa(ex, Int) || isa(ex, Char)
        return true
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return isindex(ex.args[1])
    else
        return false
    end
end

# test for a simple tensor object indexed by valid indices
function istensor(ex)
    if isa(ex, Expr) && (ex.head == :ref || ex.head == :typed_vcat)
        # check object
        if isa(ex.args[1], Symbol) ||
            (isa(ex.args[1], Expr) && ex.args[1].head == prime) # currently we only support adjoint on tensor objects
            #check indices
            if length(ex.args) >= 2 && isa(ex.args[2], Expr) && ex.args[2].head == :parameters
                return all(isindex, ex.args[2].args) && all(isindex, ex.args[3:end])
            else
                return all(isindex, ex.args[2:end])
            end
        end
    end
    return false
end

# test for an extended tensor object, possibly conjugated or multiplied by a scalar
function istensor2(ex)
    if istensor(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        return istensor2(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :*
        numtensor = 0
        for k = 2:length(args)
            if istensor2(ex.args[k])
                numtensor += 1
            else
                isscalarexpr(ex.args[k]) || return false
            end
        end
        numtensor == 1 || return false
        return true
    end
    return false
end

# test for a scalar expression, i.e. no indices
function isscalarexpr(ex::Expr)
    if ex.head == :call && ex.args[1] == :scalar
        return true
    elseif ex.head == :ref || ex.head == :typed_vcat
        return false
    else
        return all(isscalarexpr, ex.args)
    end
end
isscalarexpr(ex::Symbol) = true
isscalarexpr(ex::Number) = true

# test for a tensor expression, i.e. something that can be evaluated to a tensor
function istensorexpr(ex)
    if istensor(ex) || isscalarexpr(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        return istensorexpr(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && (ex.args[1] == :* || ex.args[1] == :+ || ex.args[1] == :-)
        return all(istensorexpr, ex.args[2:end])
    end
    return false
end

# convert an expression into a valid index
function makeindex(ex)
    if isa(ex, Symbol) || isa(ex, Int) || isa(ex, Char)
        return ex
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return Symbol(makeindex(ex.args[1]), "′")
    else
        error("not a valid index: $ex")
    end
end

# extract the tensor object itself, as well as its left and right indices
function maketensor(ex)
    if isa(ex, Expr) && (ex.head == :ref || ex.head == :typed_vcat)
        # check object
        if isa(ex.args[1], Symbol) ||
            (isa(ex.args[1], Expr) && ex.args[1].head == prime) # currently we only support adjoint
            #check indices
            object = ex.args[1]
            if length(ex.args) >= 2 && isa(ex.args[2], Expr) && ex.args[2].head == :parameters
                leftind = map(makeindex, ex.args[3:end])
                rightind = map(makeindex, ex.args[2].args)
            else
                leftind = map(makeindex, ex.args[2:end])
                rightind = Any[]
            end
            return (object, leftind, rightind)
        end
    end
    throw(ArgumentError())
end

function maketensor2(ex)
    if istensor(ex)
        obj, leftind, rightind = maketensor(ex)
        return (obj, leftind, rightind, +1, false)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        obj, leftind, rightind, scalar, conj = maketensor2(ex.args[2])
        return (obj, leftind, rightind, :(conj($scalar)), !conj)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :*

        numtensor = 0
        for k = 2:length(args)
            if istensor2(ex.args[k])
                numtensor += 1
            else
                isscalarexpr(ex.args[k]) || return false
            end
        end
        numtensor == 1 || return false
        return true
    end
    return false
end


# for any index expression, get the list of uncontracted indices from that expression
function getindices(ex::Expr)
    if ex.head == :ref || ex.head == :typed_vcat
        if length(ex.args) >= 2 && isa(ex.args[2], Expr) && ex.args[2].head == :parameter
            leftind = map(makeindex, ex.args[3:end])
            rightind = map(makeindex, ex.args[2].args)
            return unique2(vcat(leftind, rightind))
        else
            return unique2(map(makeindex, ex.args[2:end]))
        end
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

# get all unique indices appearing in the expression
function getallindices(ex::Expr)
    if ex.head == :ref || ex.head == :typed_vcat
        if length(ex.args) >= 2 && isa(ex.args[2], Expr) && ex.args[2].head == :parameter
            leftind = map(makeindex, ex.args[3:end])
            rightind = map(makeindex, ex.args[2].args)
            return unique(vcat(leftind, rightind))
        else
            return unique(map(makeindex, ex.args[2:end]))
        end
    elseif !isempty(ex.args)
        return unique(mapreduce(getallindices, vcat, ex.args))
    else
        return Vector{Any}()
    end
end
getallindices(ex) = Vector{Any}()
