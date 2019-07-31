# Test for valid tensor expressions and parse them.
const prime = Symbol("'")

# test for assignment (copy into existing tensor) or definition (create new tensor)
isassignment(ex::Expr) = ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=)
function isdefinition(ex::Expr)
    #TODO: remove when := is removed
    # if ex.head == :(:=)
    #     warn(":= will likely be deprecated as assignment operator in Julia, use ≔ (\\coloneq + TAB) or go to http://github.com/Jutho/TensorOperations.jl to suggest ASCII alternatives", once=true, key=:warnaboutcoloneq)
    # end
    return ex.head == :(:=) || ex.head == :(≔)
end

# get left and right hand side from an expression or definition
function getlhsrhs(ex::Expr)
    if ex.head in (:(=), :(+=), :(-=), :(:=), :(≔))
        return ex.args[1], ex.args[2]
    else
        error("invalid assignment or definition $ex")
    end
end

# test for a valid index
function isindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return true
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return isindex(ex.args[1])
    else
        return false
    end
end

# test for a simple tensor object indexed by valid indices
istensor(ex) = false
function istensor(ex::Expr)
    if ex.head == :ref || ex.head == :typed_hcat
        return all(isindex, ex.args[2:end])
    elseif ex.head == :typed_vcat
        length(ex.args) == 3 || return false
        if isa(ex.args[2], Expr) && ex.args[2].head == :row
            all(isindex, ex.args[2].args) || return false
        else
            isindex(ex.args[2]) || return false
        end
        if isa(ex.args[3], Expr) && ex.args[3].head == :row
            all(isindex, ex.args[3].args) || return false
        else
            isindex(ex.args[3]) || return false
        end
        return true
    end
    return false
end

# test for a generalized tensor, i.e. with scalar multiplication and conjugation
isgeneraltensor(ex) = false
function isgeneraltensor(ex::Expr)
    if istensor(ex)
        return true
    elseif ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2
        # unary plus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :- && length(ex.args) == 2
        # unary minus
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2
        # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :adjoint && length(ex.args) == 2
        # adjoint
        return isgeneraltensor(ex.args[2])
    elseif ex.head == prime && length(ex.args) == 1
        # adjoint
        return isgeneraltensor(ex.args[1])
    elseif ex.head == :call && ex.args[1] == :transpose && length(ex.args) == 2
        # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3
        # scalar multiplication
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            return true
        elseif isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            return true
        end
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        # scalar multiplication
        if isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            return true
        end
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        # scalar multiplication
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            return true
        end
    end
    return false
end

function hastraceindices(ex)
    obj,leftind, rightind, = makegeneraltensor(ex)
    allind = vcat(leftind, rightind)
    return length(allind) != length(unique(allind))
end

# test for a scalar expression, i.e. no indices
function isscalarexpr(ex::Expr)
    if ex.head == :call && ex.args[1] == :scalar
        return true
    elseif ex.head in (:ref, :typed_vcat, :typed_hcat)
        return false
    else
        return all(isscalarexpr, ex.args)
    end
end
isscalarexpr(ex::Symbol) = true
isscalarexpr(ex::Number) = true
isscalarexpr(ex) = false

# test for a tensor expression, i.e. something that can be evaluated to a tensor
function istensorexpr(ex)
    if isgeneraltensor(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        return all(istensorexpr, ex.args[2:end]) # all arguments should be tensor expressions (we are not checking matching indices yet)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :*
        return any(istensorexpr, ex.args[2:end]) # at least one argument should be a tensor expression
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        return istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        return istensorexpr(ex.args[3]) && isscalarexpr(ex.args[2])
    end
    return false
end

# convert an expression into a valid index
function makeindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return ex
    elseif isa(ex, Expr) && ex.head == prime && length(ex.args) == 1
        return Symbol(makeindex(ex.args[1]), "′")
    else
        error("not a valid index: $ex")
    end
end

# extract the tensor object itself, as well as its left and right indices
function maketensor(ex)
    if isa(ex, Expr) && (ex.head == :ref || ex.head == :typed_hcat)
        object = esc(ex.args[1])
        leftind = Any[makeindex(x) for x in ex.args[2:end]]
        rightind = Any[]
        return (object, leftind, rightind)
    elseif isa(ex, Expr) && ex.head == :typed_vcat
        length(ex.args) <= 3 || throw(ArgumentError("invalid tensor index expression: $ex"))
        object = esc(ex.args[1])
        if isa(ex.args[2], Expr) && ex.args[2].head == :row
            leftind = Any[makeindex(x) for x in ex.args[2].args]
        elseif ex.args[2] == :_
            leftind = Any[]
        else
            leftind = Any[makeindex(ex.args[2])]
        end
        if length(ex.args) > 2 && isa(ex.args[3], Expr) && ex.args[3].head == :row
            rightind = Any[makeindex(x) for x in ex.args[3].args]
        elseif length(ex.args) == 2 || ex.args[3] == :_
            rightind = Any[]
        else
            rightind = Any[makeindex(ex.args[3])]
        end
        return (object, leftind, rightind)
    end
    throw(ArgumentError())
end
# extract the tensor object itself, as well as its left and right indices
function replacetensorobj!(ex, s)
    if isa(ex, Expr) && (ex.head == :ref || ex.head == :typed_hcat)
        ex.args[1] = s
        return ex
    elseif isa(ex, Expr) && ex.head == :typed_vcat
        length(ex.args) <= 3 || throw(ArgumentError("invalid tensor index expression: $ex"))
        ex.args[1] = s
        return ex
    end
    throw(ArgumentError())
end

# extract the tensor object itself, as well as its left and right indices, its scalar factor and its conjugation flag
function makegeneraltensor(ex)
    if istensor(ex)
        object, leftind, rightind = maketensor(ex)
        return (object, leftind, rightind, 1, false)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2 # unary plus: pass on
        return makegeneraltensor(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :- && length(ex.args) == 2 # unary minus: flip scalar factor
        (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[2])
        return (object, leftind, rightind, Expr(:call, :-, α), conj)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2 # conjugation: flip conjugation flag and conjugate scalar factor
        (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[2])
        return (object, leftind, rightind, Expr(:call, :conj, α), !conj)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[3])
            return (object, leftind, rightind, Expr(:call, :*, makescalar(ex.args[2]), α), conj)
        elseif isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :*, α, makescalar(ex.args[3])), conj)
        end
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :/, α, makescalar(ex.args[3])), conj)
        end
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[3])
            return (object, leftind, rightind, Expr(:call, :\, makescalar(ex.args[2]), α), conj)
        end
    end
    throw(ArgumentError("not a valid generalized tensor expression $ex"))
end

function makescalar(ex::Expr)
    if ex.head == :call && ex.args[1] == :scalar
        @assert length(ex.args) == 2 && istensorexpr(ex.args[2])
        return :(scalar($(deindexify(nothing, 0, ex.args[2], 1, [], [], true))))
    elseif ex.head == :call
        return Expr(ex.head, ex.args[1], map(makescalar, ex.args[2:end])...)
    else
        return Expr(ex.head, map(makescalar, ex.args)...)
    end
end
makescalar(ex::Symbol) = esc(ex)
makescalar(ex) = ex

# for any tensor expression, get the list of uncontracted indices that would remain after evaluating that expression
function getindices(ex::Expr)
    if istensor(ex)
        _,leftind,rightind = maketensor(ex)
        return unique2(vcat(leftind, rightind))
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        return getindices(ex.args[2]) # getindices on any of the args[2:end] should yield the same result
    elseif ex.head == :call && ex.args[1] == :*
        indices = getindices(ex.args[2])
        for k = 3:length(ex.args)
            append!(indices, getindices(ex.args[k]))
        end
        return unique2(indices)
    elseif ex.head == :call && ex.args[1] == :/
        indices = getindices(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :\
        indices = getindices(ex.args[3])
    elseif ex.head == :call && length(ex.args) == 2
        return getindices(ex.args[2])
    else
        return Vector{Any}()
    end
end
getindices(ex) = Vector{Any}()

# get all unique indices appearing in a tensor expression
function getallindices(ex::Expr)
    if istensor(ex)
        _,leftind,rightind = maketensor(ex)
        return unique(vcat(leftind, rightind))
    elseif !isempty(ex.args)
        return unique(mapreduce(getallindices, vcat, ex.args))
    else
        return Vector{Any}()
    end
end
getallindices(ex) = Vector{Any}()

function geteltype(ex::Expr)
    if istensor(ex)
        obj,_,_ = maketensor(ex)
        return Expr(:call, :eltype, obj)
    elseif ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :- || ex.args[1] == :* || ex.args[1] == :/)
        if length(ex.args) > 2
            return Expr(:call, :promote_type, map(geteltype, ex.args[2:end])...)
        else
            return geteltype(ex.args[2])
        end
    elseif ex.head == :call && ex.args[1] == :conj
        return geteltype(ex.args[2])
    elseif isscalarexpr(ex)
        return :(typeof($ex))
    else
        # return :(eltype($ex)) # would probably lead to doing the same operation twice
        throw(ArgumentError("unable to determine eltype"))
    end
end
geteltype(ex) = Expr(:call,:typeof, esc(ex))
