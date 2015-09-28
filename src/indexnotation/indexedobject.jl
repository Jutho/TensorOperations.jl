# indexnotation/indexedobject.jl
#
# Defines a parameteric type to store indices at compile-time, as well as a
# hierarchy of types to assign these indices to objects and implement tensor
# operations with them.

import Base: +, -, *

immutable Indices{I}
end

abstract AbstractIndexedObject
indices(a::AbstractIndexedObject) = indices(typeof(a))

immutable IndexedObject{I,C,A,T} <: AbstractIndexedObject
    object::A
    α::T
    function IndexedObject(object::A, α::T)
        checkindices(object, I)
        new(object, α)
    end
end
Base.call{I,C, A, T}(::Type{IndexedObject{I,C}}, object::A, α::T=1) = IndexedObject{I,C, A, T}(object, α)

Base.conj{I}(a::IndexedObject{I,:N}) = IndexedObject{I,:C}(a.object, conj(a.α))
Base.conj{I}(a::IndexedObject{I,:C}) = IndexedObject{I,:N}(a.object, conj(a.α))

Base.eltype(A::IndexedObject) = promote_type(eltype(A.object), typeof(A.α))
Base.eltype{I,C, A, T}(::Type{IndexedObject{I,C, A, T}}) = promote_type(eltype(A), T)

*{I,C}(a::IndexedObject{I,C}, β::Number) = IndexedObject{I,C}(a.object, a.α*β)
*(β::Number, a::IndexedObject) = *(a, β)
-(a::IndexedObject) = *(a, -1)

@generated function indices{I,C,A,T}(::Type{IndexedObject{I,C,A,T}})
    J = tuple(unique2(I)...)
    meta = Expr(:meta, :inline)
    Expr(:block, :meta, :($J))
end

indexify{I}(object, ::Indices{I}) = IndexedObject{I,:N}(object)

deindexify{I}(A::IndexedObject{I,:N}, ::Indices{I}) = A.α == 1 ? A.object : A.α*A.object
deindexify{I}(A::IndexedObject{I,:C}, ::Indices{I}) = A.α == 1 ? conj(A.object) : A.α*conj(A.object)

@generated function deindexify{I,C,J}(A::IndexedObject{I,C}, ::Indices{J}, T::Type = eltype(A))
    meta = Expr(:meta, :inline)
    indCinA, = trace_indices(I,J)
    conj = Val{C}
    quote
        $meta
        deindexify!(similar_from_indices(T, $indCinA, A.object, $conj), A, Indices{$J}())
    end
end

@generated function deindexify!{Idst, Isrc, C}(dst, src::IndexedObject{Isrc, C}, ::Indices{Idst}, β=0)
    meta = Expr(:meta, :inline)
    Jdst = unique2(Idst)
    length(Jdst) == length(Idst) || throw(IndexError("left-hand side cannot have partial trace: $Idst"))
    if length(Isrc) == length(Jdst)
        indCinA = add_indices(Isrc, Idst)
        :($meta;add!(src.α, src.object, Val{C}, β, dst, $indCinA))
    else
        indCinA, cindA1, cindA2 = trace_indices(Isrc, Idst)
        return :($meta;trace!(src.α, src.object, Val{C}, β, dst, $indCinA, $cindA1, $cindA2))
    end
end
