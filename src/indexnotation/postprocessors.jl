function _flatten(ex::Expr)
    head = ex.head
    args = _flatten.(ex.args)
    if head == :block
        newargs = Any[]
        for e in args
            if e isa Expr && e.head == :block
                append!(newargs, e.args)
            else
                push!(newargs, e)
            end
        end
        return Expr(:block, newargs...)
    elseif head == :(=) && args[2] isa Expr && args[2].head == :block
        newargs = args[2].args
        newargs[end] = Expr(:(=), args[1], newargs[end])
        return Expr(:block, newargs...)
    elseif head == :call && args[1] == :scalar && args[2] isa Expr && args[2].head == :block
        newargs = args[2].args
        newargs[end] = Expr(:call, args[1], newargs[end])
        return Expr(:block, newargs...)
    else
        return Expr(head, args...)
    end
end
_flatten(e) = e

function removelinenumbernode(ex::Expr)
    if ex.head == :block
        return Expr(:block,
                    (removelinenumbernode(e) for e in ex.args if !(e isa LineNumberNode))...)
    else
        return ex
    end
end
removelinenumbernode(ex) = ex

const tensoroperationscorefunctions = (:tensoralloc, :tensoralloctemp, :tensorfree!,
                                       :tensoradd!, :tensortrace!, :tensorcontract!,
                                       :tensorscalar, :IndexError,
                                       :scalartype, :checkcontractible)
const tensoroperationsfunctions = (:promote_contract, :promote_add, :tensoralloc_add,
                                   :tensoralloc_contract)

function addtensoroperations(ex::Expr)
    if ex.head == :call && ex.args[1] in tensoroperationscorefunctions
        return Expr(ex.head, GlobalRef(TensorOperationsCore, ex.args[1]),
                    (addtensoroperations(ex.args[i]) for i in 2:length(ex.args))...)
    elseif ex.head == :call && ex.args[1] in tensoroperationsfunctions
        return Expr(ex.head, GlobalRef(TensorOperations, ex.args[1]),
                    (addtensoroperations(ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (addtensoroperations(e) for e in ex.args)...)
    end
end
addtensoroperations(ex) = ex

const operationfunctions = (:tensoradd!, :tensortrace!, :tensorcontract!)
insertoperationbackend(ex, backend) = ex
function insertoperationbackend(ex::Expr, backend)
    if ex.head == :call && ex.args[1] in operationfunctions
        insert!(ex.args, 2, backend)
        return ex
    else
        return Expr(ex.head, (insertoperationbackend(e, backend) for e in ex.args)...)
    end
end

const allocationfunctions = (:tensoralloc, :tensoralloctemp, :tensorfree!)
insertallocationbackend(ex, backend) = ex
function insertallocationbackend(ex::Expr, backend)
    if ex.head == :call && ex.args[1] in allocationfunctions
        insert!(ex.args, 2, backend)
        return ex
    else
        return Expr(ex.head, (insertallocationbackend(e, backend) for e in ex.args)...)
    end
end