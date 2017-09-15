# check if a list of indices specifies a tensor contraction in ncon style
function isnconstyle(network::Vector)
    allindices = Vector{Int}()
    for ind in network
        all(i->isa(i, Integer), ind) || return false
        append!(allindices, ind)
    end
    while length(allindices) > 0
        i = pop!(allindices)
        if i > 0 # positive labels represent contractions or traces and should appear twice
            k = findnext(allindices, i, 1)
            l = findnext(allindices, i, k+1)
            if k == 0 || l != 0
                return false
            end
            deleteat!(allindices, k)
        elseif i < 0 # negative labels represent open indices and should appear once
            findnext(allindices, i, 1) == 0 || return false
        else # i == 0
            return false
        end
    end
    return true
end

function ncontree(network::Vector)
    contractionindices = Vector{Vector{Int}}(length(network))
    for k = 1:length(network)
        indices = network[k]
        # trace indices have already been removed, remove open indices by filtering on positive values
        contractionindices[k] = filter(i->i>0, indices)
    end
    partialtrees = collect(Any, 1:length(network))
    _ncontree!(partialtrees, contractionindices)
end

function _ncontree!(partialtrees, contractionindices)
    if length(partialtrees) == 1
        return partialtrees[1]
    end
    if all(isempty, contractionindices) # disconnected network
        partialtrees[end-1] = (partialtrees[end-1], partialtrees[end])
        pop!(partialtrees)
        pop!(contractionindices)
    else
        let firstind = minimum(flatten(contractionindices))
            i1 = findfirst(x->in(firstind,x), contractionindices)
            i2 = findnext(x->in(firstind,x), contractionindices, i1+1)
            newindices = unique2(vcat(contractionindices[i1], contractionindices[i2]))
            newtree = (partialtrees[i1], partialtrees[i2])
            partialtrees[i1] = newtree
            deleteat!(partialtrees, i2)
            contractionindices[i1] = newindices
            deleteat!(contractionindices, i2)
        end
    end
    _ncontree!(partialtrees, contractionindices)
end
