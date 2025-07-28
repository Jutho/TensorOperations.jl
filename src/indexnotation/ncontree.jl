# Verify if a list of indices specifies a tensor contraction in ncon style.
check_nconstyle(::Type{Bool}, network) = _check_nconstyle(network, Val(true))
check_nconstyle(network) = _check_nconstyle(network, Val(false))

function _check_nconstyle(network, ::Val{check}) where {check}
    allindices = Vector{Int}()
    for ind in network
        all(i -> isa(i, Integer), ind) || return check ? false :
            throw(IndexError("All indices must be integers"))
        append!(allindices, ind)
    end
    while length(allindices) > 0
        i = pop!(allindices)
        if i > 0 # positive labels represent contractions or traces and should appear twice
            k = findfirst(isequal(i), allindices)
            isnothing(k) && return check ? false :
                throw(IndexError(lazy"Index $i appears only once in the network"))
            l = findnext(isequal(i), allindices, k + 1)
            !isnothing(l) && return check ? false :
                throw(IndexError(lazy"Index $i appears more than twice in the network"))
            deleteat!(allindices, k)
        elseif i < 0 # negative labels represent open indices and should appear once
            isnothing(findfirst(isequal(i), allindices)) || return check ? false :
                throw(IndexError(lazy"Index $i appears more than once in the network"))
        else # i == 0
            return check ? false : throw(IndexError("Index 0 is not allowed in the network"))
        end
    end
    return check ? true : nothing
end

function ncontree(network)
    contractionindices = Vector{Vector{Int}}(undef, length(network))
    for k in 1:length(network)
        indices = network[k]
        # trace indices have already been removed, remove open indices by filtering on positive values
        contractionindices[k] = filter(i -> i > 0, indices)
    end
    partialtrees = collect(Any, 1:length(network))
    return _ncontree!(partialtrees, contractionindices)
end

function _ncontree!(partialtrees, contractionindices)
    if length(partialtrees) == 1
        return partialtrees[1]
    end
    if all(isempty, contractionindices) # disconnected network
        partialtrees[end - 1] = Any[partialtrees[end - 1], partialtrees[end]]
        pop!(partialtrees)
        pop!(contractionindices)
    else
        let firstind = minimum(vcat(contractionindices...))
            i1 = findfirst(x -> in(firstind, x), contractionindices)
            i2 = findnext(x -> in(firstind, x), contractionindices, i1 + 1)
            @assert i1 !== nothing && i2 !== nothing
            newindices = unique2(vcat(contractionindices[i1], contractionindices[i2]))
            newtree = Any[partialtrees[i1], partialtrees[i2]]
            partialtrees[i1] = newtree
            deleteat!(partialtrees, i2)
            contractionindices[i1] = newindices
            deleteat!(contractionindices, i2)
        end
    end
    return _ncontree!(partialtrees, contractionindices)
end

"""
    nconindexcompletion(ex)

Complete the indices of the left hand side of an ncon expression. For example, the following expressions are equivalent after index completion.

    @tensor A[:] := B[-1, 1, 2] * C[1, 2, -3]
    @tensor A[-1, -2] := B[-1, 1, 2] * C[1, 2, -3]
"""
function nconindexcompletion(ex)
    if isassignment(ex) || isdefinition(ex)
        lhs, rhs = getlhs(ex), getrhs(ex)
        # process left hand side
        if istensor(lhs) && istensorexpr(rhs)
            indices = getindices(rhs)

            if lhs.head == :ref && length(lhs.args) == 2 && lhs.args[2] == :(:)
                if all(isa(i, Integer) && i < 0 for i in indices)
                    lhs = Expr(:ref, lhs.args[1], sort(indices; rev = true)...)
                else
                    error("cannot automatically infer index order of left hand side")
                end
            end
            return Expr(ex.head, lhs, rhs)
        else
            return ex
        end
    elseif ex isa Expr
        return Expr(ex.head, map(nconindexcompletion, ex.args)...)
    else
        return ex
    end
end

function resolve_traces(tensors, network)
    transformed = map(zip(tensors, network)) do (A, IA)
        IC = unique2(IA)
        return length(IC) == length(IA) ? (A, IA) : (tensortrace(IC, A, IA), IC)
    end
    return first.(transformed), last.(transformed)
end
