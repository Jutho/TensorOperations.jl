
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
        TType, structure = tensorstructure(obj)

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

struct TensorCache <: Backend end

function TOC.tensoralloc(::TensorCache, args...)
    return tensor_from_structure(tensorstructure(args...)...)
end

function TOC.tensoralloctemp(::TensorCache, args...)
    TType, str = tensorstructure(args...)
    return allocate(GlobalPool, TType, str)::TType
end

TOC.tensorfree!(::TensorCache, obj) = deallocate!(GlobalPool, obj)

# generic definitions, should be overwritten if your array/tensor type does not support
# Base.similar(object, eltype, structure)
function similar_from_structure(A, T, structure)
    if isbitstype(T)
        similar(A, T, structure)
    else
        fill!(similar(A, T, structure), zero(T)) # this fixes BigFloat issues
    end
end

function similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol)
    structure = similarstructure_from_indices(T, p1, p2, A, CA)
    return similar_from_structure(A, T, structure)
end
function similar_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
                              p1::IndexTuple, p2::IndexTuple,
                              A, B, CA::Symbol, CB::Symbol)
    structure = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
    return similar_from_structure(A, T, structure)
end

# should work generically but can be overwritten
function similartype_from_indices(T::Type, p1, p2, A, CA)
    return Core.Compiler.return_type(similar_from_indices,
                                     Tuple{Type{T},typeof(p1),typeof(p2),typeof(A),Symbol})
end
function similartype_from_indices(T::Type, poA, poB, p1, p2, A, B, CA, CB)
    return Core.Compiler.return_type(similar_from_indices,
                                     Tuple{Type{T},typeof(poA),typeof(poB),
                                           typeof(p1),typeof(p2),typeof(A),typeof(B),
                                           Symbol,Symbol})
end

# generic, should probably not be overwritten
function cached_similar_from_indices(sym::Symbol, T::Type,
                                     p1::IndexTuple, p2::IndexTuple,
                                     A, CA::Symbol)
    if use_cache()
        structure = similarstructure_from_indices(T, p1, p2, A, CA)
        typ = similartype_from_indices(T, p1, p2, A, CA)
        key = (sym, taskid(), typ, structure)
        C::typ = get!(cache, key) do
            return similar_from_indices(T, p1, p2, A, CA)
        end
        return C
    else
        return similar_from_indices(T, p1, p2, A, CA)
    end
end
function cached_similar_from_indices(sym::Symbol, T::Type,
                                     poA::IndexTuple, poB::IndexTuple,
                                     p1::IndexTuple, p2::IndexTuple,
                                     A, B, CA::Symbol, CB::Symbol)
    if use_cache()
        structure = similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        typ = similartype_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        key = (sym, taskid(), typ, structure)
        C::typ = get!(cache, key) do
            return similar_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
        end
        return C
    else
        return similar_from_indices(T, poA, poB, p1, p2, A, B, CA, CB)
    end
end
