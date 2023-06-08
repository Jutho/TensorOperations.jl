# remove nested structure of expression, returning a flat line of expressions
function _flatten(ex)
    if isa(ex, Expr) # prewalk
        ex = Expr(ex.head, map(_flatten, ex.args)...)
    end
    if isexpr(ex, :block)
        newargs = Any[]
        for e in ex.args
            if e isa Expr && e.head == :block
                append!(newargs, e.args)
            else
                push!(newargs, e)
            end
        end
        return Expr(:block, newargs...)
    elseif isexpr(ex, :(=)) && isexpr(ex.args[2], :block)
        newargs = ex.args[2].args
        newargs[end] = Expr(:(=), ex.args[1], newargs[end])
        return Expr(:block, newargs...)
    elseif head == :call && args[1] == :tensorscalar && args[2] isa Expr &&
           args[2].head == :block
        newargs = args[2].args
        newargs[end] = Expr(:call, args[1], newargs[end])
        return Expr(:block, newargs...)
    else
        return ex
    end
end

# remove line number nodes
function removelinenumbernode(ex)
    if isexpr(ex, :block)
        args = [removelinenumbernode(e) for e in ex.args if !(e isa LineNumberNode)]
        return Expr(:block, args...)
    else
        return ex
    end
end

# fix referene to TensorOperation functions
const tensoroperationsfunctions = (:tensoralloc, :tensoralloctemp, :tensorfree!,
                                   :tensoradd!, :tensortrace!, :tensorcontract!,
                                   :tensorscalar, :IndexError, :scalartype,
                                   :checkcontractible, :promote_contract, :promote_add,
                                   :tensoralloc_add, :tensoralloc_contract)
function addtensoroperations(ex)
    if isexpr(ex, :call) && ex.args[1] in tensoroperationsfunctions
        return Expr(ex.head, GlobalRef(TensorOperations, ex.args[1]),
                    (addtensoroperations(ex.args[i]) for i in 2:length(ex.args))...)
    elseif isa(ex, Expr)
        return Expr(ex.head, (addtensoroperations(e) for e in ex.args)...)
    else
        return ex
    end
end