"""
    _flatten(ex)

Flatten nested structure of an expression, returning a flat line of expressions.
"""
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
    elseif isexpr(ex, :call) && ex.args[1] == :tensorscalar && isexpr(ex.args[2], :block)
        newargs = ex.args[2].args
        newargs[end] = Expr(:call, ex.args[1], newargs[end])
        return Expr(:block, newargs...)
    else
        return ex
    end
end

"""
    removelinenumbernode(ex)

Remove all `LineNumberNode`s from an expression.
"""
function removelinenumbernode(ex)
    if isexpr(ex, :block)
        args = [removelinenumbernode(e) for e in ex.args if !(e isa LineNumberNode)]
        return Expr(:block, args...)
    else
        return ex
    end
end

"list of functions that are used in expressions produced by `@tensor`"
const tensoroperationsfunctions = (:tensoralloc, :tensorfree!,
                                   :tensoradd!, :tensortrace!, :tensorcontract!,
                                   :tensorscalar, :tensorcost, :IndexError, :scalartype,
                                   :checkcontractible, :promote_contract, :promote_add,
                                   :tensoralloc_add, :tensoralloc_contract)
"""
    addtensoroperations(ex)

Fix references to TensorOperations functions in namespaces where `@tensor` is present but the functions are not.
"""
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

const backend_operations = (:tensoradd!, :tensorcontract!, :tensortrace!, :tensoralloc_add,
                            :tensoralloc_contract)

"""
    insertbackend(ex, backend)

Insert a backend into a tensor operation, e.g.
- `tensoradd!(args...)` -> `tensoradd!(args..., Backend{:backend}())`
- `tensorcontract!(args...)` -> `tensorcontract!(args..., Backend{:backend}())`
- `tensortrace!(args...)` -> `tensortrace!(args..., Backend{:backend}())`
- `tensoralloc_add(args...)` -> `tensoralloc_add(args..., Backend{:backend}())`
- `tensoralloc_contract(args...)` -> `tensoralloc_contract(args..., Backend{:backend}())`
"""
function insertbackend(ex, backend)
    if isexpr(ex, :call) && ex.args[1] isa GlobalRef &&
       ex.args[1].mod == TensorOperations &&
       ex.args[1].name ∈ backend_operations
        b = Backend{backend}()
        return Expr(:call, ex.args..., b)
    elseif isa(ex, Expr)
        return Expr(ex.head, (insertbackend(e, backend) for e in ex.args)...)
    else
        return ex
    end
end
