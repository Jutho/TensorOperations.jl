
# ---------------------------------------------------------------------------------------- #
# Cache implementation
# ---------------------------------------------------------------------------------------- #

struct LinkedNode{T}
    next::Union{LinkedNode,Nothing}
    val::T
end

mutable struct ObjectPool{K,V}
    pool::ConcurrentDict{K,V}
    @atomic currentsize::Int
    maxsize::Int
    ObjectPool{K,V}(maxsize::Int) where {K,V} = new{K,V}(ConcurrentDict{K,V}(), 0, maxsize)
end

ObjectPool(maxsize::Int) = ObjectPool{Any,Any}(maxsize)

# generic definition, net very efficient, provide more efficient version if possible
memsize(a::Any) = Base.summarysize(a)

modify!(f, pool::ObjectPool, key) = ConcurrentCollections.modify!(f, pool.pool, key)

# request an object from a pool, or allocate a new object
function allocate(objpool::ObjectPool, TType, structure)
    allocated::Ref{TType} = Ref{TType}()
    is_set::Bool = false

    # key is not in pool
    function pop_create(::Nothing)
        is_set = false
        return Some(nothing)
    end

    # key is in pool
    function pop_create(ref)
        stack = ref[]
        if !isnothing(stack)
            allocated = Ref(stack.val)
            is_set = true
            return Some(stack.next)
        else
            is_set = false
            return Some(stack)
        end
    end

    modify!(pop_create, objpool, (TType, structure))

    if is_set
        toret::TType = allocated[]
        @atomic objpool.currentsize -= memsize(toret)
        return toret
    else
        return tensor_from_structure(TType, structure)::TType
    end
end

# free an object by returning it to a pool
function deallocate!(objpool::ObjectPool, obj)
    let obj = obj
        TType, structure = typeof(obj), tensorstructure(obj)

        cs = @atomic objpool.currentsize += memsize(obj)
        cs > objpool.maxsize && unsafe_process!(objpool)

        pop_create(::Nothing) = Some(LinkedNode(nothing, obj))
        pop_create(val) = Some(LinkedNode(val[], obj))
        modify!(pop_create, objpool, (TType, structure))
    end
end

# clean up overfull pool
function unsafe_process!(objp)
    # iterate over slots, free up every pool
    for k in keys(objp.pool)
        y = modify!(objp, k) do x
            isnothing(x) && return nothing
            return Delete(x)
        end

        if !isnothing(y)
            node = y.value[]
            while !isnothing(node)
                @atomic objp.currentsize -= memsize(node.val)
                node = node.next
            end
        end
    end
end

# ---------------------------------------------------------------------------------------- #
# Pool options
# ---------------------------------------------------------------------------------------- #

function default_cache_size()
    return min(1 << 32, Int(Sys.total_memory()) >> 2)
end
const GlobalPool = ObjectPool(default_cache_size())

cachesize() = GlobalPool.currentsize

# function TOC.tensoralloc(::CacheBackend, args...)
#     return tensor_from_structure(tensorstructure(args...)...)
# end

# function TOC.tensoralloctemp(::CacheBackend, args...)
#     TType, str = tensorstructure(args...)
#     return allocate(GlobalPool, TType, str)::TType
# end

TOC.tensorfree!(::CacheBackend, obj) = deallocate!(GlobalPool, obj)
