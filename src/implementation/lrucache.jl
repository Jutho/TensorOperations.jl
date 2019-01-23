# Taken and adapted from LRUCache.jl
mutable struct LRUNode{K, V}
    k::K
    v::V
    next::LRUNode{K, V}
    prev::LRUNode{K, V}

    # All new created nodes are self referential only
    function LRUNode{K, V}(k::K, v::V) where {K, V}
        x = new{K, V}(k, v)
        x.next = x
        x.prev = x
        return x
    end
end

mutable struct LRUList{K, V}
    first::Union{LRUNode{K, V}, Nothing}
    size::Int64

    LRUList{K, V}() where {K, V} = new{K, V}(nothing, 0)
end

Base.first(l::LRUList) = !isempty(l) ? l.first : error("LRUList is empty")
Base.last(l::LRUList) = !isempty(l) ? l.first.prev : error("LRUList is empty")

Base.length(l::LRUList) = l.size
Base.isempty(l::LRUList) = length(l) == 0

function Base.show(io::IO, l::LRUNode{K, V}) where {K, V}
    print(io, "LRUNode{", K, ", ", V, "}(")
    show(io, l.k)
    print(io, ", ")
    show(io, l.v)
    print(io, ")")
end

function Base.show(io::IO, l::LRUList{K, V}) where {K, V}
    print(io, "LRUList{", K, ", ", V, "}(")
    if length(l) != 0
        f = first(l)
        show(io, f.k)
        print(io, "=>")
        show(io, f.v)
        n = f.next
        while n !== f
            print(io, ", ")
            show(io, n.k)
            print(io, "=>")
            show(io, n.v)
            n = n.next
        end
    end
    print(io, ")")
end

function Base.push!(list::LRUNode{K, V}, new::LRUNode{K, V}) where {K, V}
    new.next = list
    new.prev = list.prev
    list.prev.next = new
    list.prev = new
    return list
end

function Base.push!(l::LRUList{K, V}, el::LRUNode{K, V}) where {K, V}
    if isempty(l)
        l.first = el
    else
        push!(l.first, el)
    end
    l.size += 1
    return l
end

function Base.pop!(l::LRUList{K, V}, n::LRUNode{K, V}=last(l)) where {K, V}
    if n.next === n
        l.first = nothing
    else
        if n === first(l)
            l.first = n.next
        end
        n.next.prev = n.prev
        n.prev.next = n.next
    end
    l.size -= 1
    return n
end

function Base.pushfirst!(l::LRUList{K, V}, el::LRUNode{K, V}) where {K, V}
    push!(l, el)
    rotate!(l)
end

# Rotate one step forward, so last element is now first
function rotate!(l::LRUList)
    if length(l) > 1
        l.first = first(l).prev
    end
    return l
end

# Move the node n to the front of the list
function move_to_front!(l::LRUList{T}, n::LRUNode{T}) where {T}
    if first(l) !== n
        pop!(l, n)
        pushfirst!(l, n)
    end
    return l
end

function Base.delete!(l::LRUList{K, V}, n::LRUNode{K, V}) where {K, V}
    pop!(l, n)
    return l
end

function Base.empty!(l::LRUList{K, V}) where {K, V}
    l.first = nothing
    l.size = 0
    return l
end

# Default cache size
const __maxfraction__ = 0.5

mutable struct LRU{K,V} <: AbstractDict{K,V}
    ht::Dict{K, LRUNode{K, V}}
    hts::Dict{K, Int64}
    q::LRUList{K, V}
    currentsize::Int64
    maxsize::Int64

    LRU{K, V}(m::Integer) where {K, V} = new{K, V}(Dict{K, V}(), Dict{K, Int64}(), LRUList{K, V}(), 0, m)
end
function LRU{K, V}(; maxsize::Integer = 0, maxrelsize::Real = 0) where {K, V}
    if maxrelsize == 0 && maxsize == 0
        m = floor(Int64, __maxfraction__ * Sys.total_memory())
    else
        m = max(maxsize, floor(Int64, maxrelsize*Sys.total_memory()))
    end
    LRU{K,V}(m)
end
LRU(; kwargs...) = LRU{Any, Any}(; kwargs...)

Base.show(io::IO, lru::LRU{K, V}) where {K, V} = print(io,"LRU{$K, $V}($(lru.maxsize))")

function Base.iterate(lru::LRU, state...)
    next = iterate(lru.ht, state...)
    if next === nothing
        return nothing
    else
        (k, node), state = next
        return k=>node.v, state
    end
end

Base.length(lru::LRU) = length(lru.q)
Base.isempty(lru::LRU) = isempty(lru.q)
function Base.sizehint!(lru::LRU, n::Integer)
    sizehint!(lru.ht, n)
    sizehint!(lru.hts, n)
    return lru
end

Base.haskey(lru::LRU, key) = haskey(lru.ht, key)
Base.get(lru::LRU, key, default) = haskey(lru, key) ? lru[key] : default

function Base.get!(default::Base.Callable, lru::LRU{K, V}, key::K) where {K,V}
    if haskey(lru, key)
        return lru[key]
    else
        value = default()
        lru[key] = value
        return value
    end
end

function Base.get!(lru::LRU{K,V}, key::K, default::V) where {K,V}
    if haskey(lru, key)
        return lru[key]
    else
        lru[key] = default
        return default
    end
end

function Base.getindex(lru::LRU, key)
    node = lru.ht[key]
    move_to_front!(lru.q, node)
    return node.v
end

function Base.setindex!(lru::LRU{K, V}, v, key) where {K, V}
    if haskey(lru, key)
        item = lru.ht[key]
        lru.currentsize -= lru.hts[key]
        item.v = v
        s = Base.summarysize(v)
        lru.currentsize += s
        lru.hts[key] = s
        move_to_front!(lru.q, item)
    else
        item = LRUNode{K, V}(key, v)
        pushfirst!(lru.q, item)
        lru.ht[key] = item
        s = Base.summarysize(v)
        lru.currentsize += s
        lru.hts[key] = s
    end
    while lru.currentsize > lru.maxsize
        rm = pop!(lru.q)
        lru.currentsize -= lru.hts[rm.k]
        delete!(lru.ht, rm.k)
        delete!(lru.hts, rm.k)
    end
    return lru
end

function setsize!(lru::LRU; maxsize::Integer = 0, maxrelsize::Real = 0)
    @assert 0 <= maxsize
    @assert 0 <= maxrelsize < 1
    if maxrelsize == 0 && maxsize == 0
        m = floor(Int64, __maxfraction__ * Sys.total_memory())
    else
        m = max(maxsize, floor(Int64, maxrelsize*Sys.total_memory()))
    end
    lru.maxsize = m
    while lru.currentsize > lru.maxsize
        rm = pop!(lru.q)
        lru.currentsize -= lru.hts[rm.k]
        delete!(lru.ht, rm.k)
        delete!(lru.hts, rm.k)
    end
end

function Base.delete!(lru::LRU, key)
    item = lru.ht[key]
    lru.currentsize -= lru.hts[key]
    delete!(lru.q, item)
    delete!(lru.ht, key)
    delete!(lru.hts, key)
    return lru
end

function Base.empty!(lru::LRU)
    lru.currentsize = 0
    empty!(lru.q)
    empty!(lru.ht)
    empty!(lru.hts)
end
