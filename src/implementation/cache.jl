using ConcurrentCollections, Base.Threads;

#=
version 2:

max_memory vs current_memory
trim all nodes when current_memory > max_memory
=#

struct LinkedNode
    next::Union{LinkedNode,Nothing}
    
    val
end

mutable struct ObjectPool
    pool::ConcurrentDict{Any,Any}

    @atomic cursize::Int # this is the amount of bytes the objectpool is currently freezing up (not in use)    
    maxsize::Int 

    ObjectPool(maxsize::Int) = new(ConcurrentDict{Any,Any}(),0,maxsize);
end

function allocate(objp::ObjectPool,type,str)
    allocated::Ref{type} = Ref{type}();
    is_set::Bool = false;

    # the key does not exist. 
    function pop_create(::Nothing)
        is_set = false;
        Some(nothing);
    end

    # the key exists, we want to attempt popping a value of the stack
    function pop_create(ref)
        stack = ref[];
        if !isnothing(stack)
            allocated = Ref(stack.val);
            is_set=true;
            Some(stack.next)
        else
            is_set=false;
            Some(stack)
        end
    end
    
    modify!(pop_create,objp.pool,(type,str));

    if is_set
        toret = allocated[];
        @atomic objp.cursize -= memorysize(toret)
        return toret::type
    else
        return allocate(type,str)::type
    end
end

function deallocate!(objp::ObjectPool,obj::type) where type
    let obj=obj
            
        str = structure(obj);
        
        cs = @atomic objp.cursize += memorysize(obj)
        
        cs > objp.maxsize && unsafe_process!(objp)

        pop_create(::Nothing) = Some(LinkedNode(nothing,obj))
        pop_create(val) = Some(LinkedNode(val[],obj))

        modify!(pop_create,objp.pool,(type,str));
    end
end

function unsafe_process!(objp)
    cur::Ref{LinkedNode} = Ref{LinkedNode}();
    is_set::Bool = false;

    function cleanup(val::Nothing)
        is_set = false;
        nothing;
    end

    
    function cleanup(val)
        cur = val
        is_set = true;
        nothing;
    end

    # iterate over slots, free up every pool
    for (k,v) in objp.pool
        modify!(cleanup,objp.pool,k);
        
        if is_set
            node = cur[];
            while is_set
                @atomic objp.cursize -= memorysize(node.val)
                isnothing(node.next) && break;
                node = node.next
            end
        end
    end
end

#-----------------
const GlobPool = ObjectPool(4*1024*1024*1024);

# stupid placeholder for the factory
allocate(::Type{T},str) where T <: AbstractTensorMap = TensorMap(undef,eltype(T),str);
function memorysize(a::AbstractTensorMap)
    sizeof(eltype(a))*sum(prod.(size.(values(blocks(a)))))
end
structure(str) = codomain(str)←domain(str);


struct DerpCache <: TensorOperations.AbstractStrategy
    pool::ObjectPool
end

TensorOperations.change_strategy(DerpCache(GlobPool));

TensorOperations.allocate_similar_from_indices(strategy::DerpCache, T::Type, poA, poB,
    p1, p2, A::AbstractArray, B::AbstractArray, CA::Symbol, CB::Symbol) = 
    TensorOperations.allocate_similar_from_indices(TensorOperations.Julia_Managed_Temporaries(), T, poA, poB, p1, p2, A, B, CA, CB)

    
TensorOperations.allocate_similar_from_indices(strategy::DerpCache,T::Type, p1, p2, A::AbstractArray, CA) = 
    TensorOperations.allocate_similar_from_indices(TensorOperations.Julia_Managed_Temporaries(), T, p1, p2, A, CA)

function TensorOperations.allocate_similar_from_indices(strategy::DerpCache, T::Type, p1, p2, A::AbstractTensorMap, CA)
    structure = TensorOperations.similarstructure_from_indices(T, p1, p2, A, CA);
    type = tensormaptype(spacetype(A),length(p1),length(p2),TensorKit.similarstoragetype(A,T));
    allocate(strategy.pool,type,structure);
end
function TensorOperations.allocate_similar_from_indices(strategy::DerpCache, T::Type, poA, poB,
    p1, p2, A::AbstractTensorMap, B::AbstractTensorMap, CA::Symbol, CB::Symbol)
    
    structure = TensorOperations.similarstructure_from_indices(T, poA, poB, p1, p2, A, B, CA, CB);
    type = tensormaptype(spacetype(A),length(p1),length(p2),TensorKit.similarstoragetype(A,T));
    allocate(strategy.pool,type,structure);
end

TensorOperations.deallocate!(strategy::DerpCache,var::AbstractTensorMap) = deallocate!(strategy.pool,var);
TensorOperations.deallocate!(strategy::DerpCache,var::AbstractArray) = nothing;