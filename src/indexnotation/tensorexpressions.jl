# Test for valid tensor expressions and parse them.
const prime = Symbol("'")

# test for assignment (copy into existing tensor) or definition (create new tensor)
isassignment(ex::Expr) = ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=)
function isdefinition(ex::Expr)
    #TODO: remove when := is removed
    # if ex.head == :(:=)
    #     warn(":= will likely be deprecated as assignment operator in Julia, use ≔ (\\coloneq + TAB) or go to http://github.com/Jutho/TensorOperations.jl to suggest ASCII alternatives", once=true, key=:warnaboutcoloneq)
    # end
    return ex.head == :(:=) || (ex.head == :call && ex.args[1] == :(≔))
end

# get left and right hand side from an expression or definition
function getlhsrhs(ex::Expr)
    if ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=) || ex.head == :(:=)
        return ex.args[1], ex.args[2]
    elseif ex.head == :call && ex.args[1] == :(≔)
        return ex.args[2], ex.args[3]
    else
        error("invalid assignment or definition $ex")
    end
end

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

# test for a generalized tensor, i.e. with scalar multiplication and conjugation
function isgeneraltensor(ex)
    if istensor(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :+ && length(ex.args) == 2 # unary plus
        return isgeneraltensor(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :- && length(ex.args) == 2 # unary minus
        return isgeneraltensor(ex.args[2])
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :conj && length(ex.args) == 2 # conjugation
        return isgeneraltensor(ex.args[2])
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # scalar multiplication
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            return true
        elseif isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            return true
        end
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
    if isgeneraltensor(ex)
        return true
    elseif isa(ex, Expr) && ex.head == :call && (ex.args[1] == :+ || ex.args[1] == :-)
        return all(istensorexpr, ex.args[2:end]) # all arguments should be tensor expressions (we are not checking matching indices yet)
    elseif isa(ex, Expr) && ex.head == :call && ex.args[1] == :*
        return any(istensorexpr, ex.args[2:end]) # at least one argument should be a tensor expression
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
            object = esc(ex.args[1])
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
        return (object, leftind, rightind, Expr(:call, esc(:conj), α), !conj)
    elseif ex.head == :call && ex.args[1] == :* && length(ex.args) == 3 # scalar multiplication: muliply scalar factors
        if isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[3])
            return (object, leftind, rightind, Expr(:call, :*, makescalar(ex.args[2]), α), conj)
        elseif isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2])
            (object, leftind, rightind, α, conj) = makegeneraltensor(ex.args[2])
            return (object, leftind, rightind, Expr(:call, :*, α, makescalar(ex.args[3])), conj)
        end
    end
    throw(ArgumentError())
end

function makescalar(ex::Expr)
    if ex.head == :call && ex.args[1] == :scalar
        @assert length(ex.args) == 2 && istensorexpr(ex.args[2])
        return :(scalar($(deindexify(ex.args[2],(),()))))
    else
        return Expr(ex.head, map(makescalar, ex.args)...)
    end
end
makescalar(ex::Symbol) = esc(ex)
makescalar(ex) = ex

# for any tensor expression, get the list of uncontracted indices that would remain after evaluating that expression
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

# get all unique indices appearing in a tensor expression
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
