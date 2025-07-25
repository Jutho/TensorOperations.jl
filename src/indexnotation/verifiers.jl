# Verifiers: tools to verify if an expression is a valid tensor expression of certain type

# test for a valid index
const prime = Symbol("'")

"""
    isindex(ex)

Test for a valid index, namely a symbol or integer, or an expression of the form `i′` where `i` is itself a valid index.
"""
function isindex(ex)
    if isa(ex, Symbol) || isa(ex, Int)
        return true
    elseif isexpr(ex, prime) && length(ex.args) == 1
        return isindex(ex.args[1])
    else
        return false
    end
end

# test for a simple tensor object indexed by valid indices
"""
    istensor(ex)

Test for a simple tensor object indexed by valid indices. This means an expression of the form:
    
    A[i, j, k, ...]
    A[i j k ...]
    A[i j k ...; l m ...]
    A[(i, j, k, ...); (l, m, ...)]
    
where `i`, `j`, `k`, ... are valid indices.
"""
function istensor end
istensor(ex) = false
function istensor(ex::Expr)
    if ex.head == :ref || ex.head == :typed_hcat
        if length(ex.args) == 1
            return true
        elseif isexpr(ex.args[2], :parameters)
            return all(isindex, ex.args[2].args) && all(isindex, ex.args[3:end])
        else
            return all(isindex, ex.args[2:end])
        end
    elseif ex.head == :typed_vcat && length(ex.args) == 3
        if isexpr(ex.args[2], [:row, :tuple])
            all(isindex, ex.args[2].args) || return false
        else
            isindex(ex.args[2]) || return false
        end
        if isexpr(ex.args[3], [:row, :tuple])
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
    elseif ex.head == :call && ex.args[1] == :*
        # scalar multiplication
        count = 0
        for i in 2:length(ex.args)
            if isgeneraltensor(ex.args[i])
                count += 1
            elseif !isscalarexpr(ex.args[i])
                return false
            end
        end
        return count == 1
    elseif ex.head == :call && ex.args[1] == :/ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[3]) && isgeneraltensor(ex.args[2]))
    elseif ex.head == :call && ex.args[1] == :\ && length(ex.args) == 3
        # scalar multiplication
        return (isscalarexpr(ex.args[2]) && isgeneraltensor(ex.args[3]))
        # TODO: disable these operations?
    elseif ex.head == :call && ex.args[1] == :adjoint && length(ex.args) == 2
        # adjoint
        return isgeneraltensor(ex.args[2])
    elseif ex.head == prime && length(ex.args) == 1
        # adjoint
        return isgeneraltensor(ex.args[1])
        # elseif ex.head == :call && ex.args[1] == :transpose && length(ex.args) == 2
        #     # transposition
        #     return isgeneraltensor(ex.args[2])
    end
    return false
end

function hastraceindices(ex)
    obj, leftind, rightind, = decomposegeneraltensor(ex)
    allind = vcat(leftind, rightind)
    return length(allind) != length(unique(allind))
end

# test for a scalar expression, i.e. no indices
"""
    isscalarexpr(ex)
    
Test for a scalar expression, i.e. an expression that can be evaluated to a scalar.
"""
function isscalarexpr(ex)
    if ex isa Symbol || ex isa Number
        return true
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar
        return istensorexpr(ex.args[2])
    elseif isexpr(ex, (:ref, :typed_vcat, :typed_hcat))
        return false
    elseif isexpr(ex, :call) # || isdefinition(ex) || isassignment(ex)
        return all(isscalarexpr, ex.args[2:end])
    else
        return true # assume everything else is valid scalar code
    end
end

# test for a tensor contraction expression
function istensorcontraction(ex)
    if isexpr(ex, :call) && ex.args[1] == :*
        return count(istensorexpr, ex.args[2:end]) >= 2
    end
    return false
end

"""
    istensorexpr(ex)

Test for a tensor expression. This means an expression which can be evaluated to a valid
    tensor. This includes:

    A[...] + B[...] - C[...] - ...
    A[...] * B[...] * ...
    λ * A[...] / μ
    λ \\ conj(A[...])
    A[...]' + adjoint(B[...]) - ...
"""
function istensorexpr(ex)
    isgeneraltensor(ex) && return true
    if isexpr(ex, :call)
        if (ex.args[1] == :+ || ex.args[1] == :-)
            return all(istensorexpr, ex.args[2:end]) # all arguments should be tensor expressions (we are not checking matching indices yet)
        elseif ex.args[1] == :*
            count = 0
            for i in 2:length(ex.args)
                if istensorexpr(ex.args[i])
                    count += 1
                elseif !isscalarexpr(ex.args[i])
                    return false
                end
            end
            return count > 0
        elseif ex.args[1] == :/ && length(ex.args) == 3
            return istensorexpr(ex.args[2]) && isscalarexpr(ex.args[3])
        elseif ex.args[1] == :\ && length(ex.args) == 3
            return istensorexpr(ex.args[3]) && isscalarexpr(ex.args[2])
        elseif ex.args[1] == :conj && length(ex.args) == 2
            return istensorexpr(ex.args[2])
        end
    end
    # TODO: disable these?
    if isexpr(ex, :call) && ex.args[1] == :adjoint && length(ex.args) == 2
        return istensorexpr(ex.args[2])
    elseif isexpr(ex, prime)
        return istensorexpr(ex.args[1])
    end
    return false
end

"""
    isassignment(ex)

Test if `ex` is an assignment expression, i.e. `ex` is of one of the forms:

    a = b
    a += b
    a -= b
"""
isassignment(ex) = isexpr(ex, [:(=), :(+=), :(-=)])

"""
    isdefinition(ex)

Test if `ex` is a definition expression, i.e. `ex` is of the form:

    a := b
    a ≔ b
"""
isdefinition(ex) = isexpr(ex, [:(:=), :(≔)])
