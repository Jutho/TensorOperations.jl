# Elementary object: IndexedObject parameterized by its indices, whether or not
# it has been conjugated, the type of the object it wraps, and the type of a
# possible scalar factor α
immutable IndexedObject{I,C,A,T} <: AbstractIndexedObject
    object::A
    α::T
    function IndexedObject(object::A,α::T)
        checklabellength(A,I)
        new(A,α)
    end
end
Base.call{I,C,A,T}(::Type{IndexedObject{I,C}},object::A,α::T=1) = IndexedObject{I,C,A,T}(object,α)

Base.conj{I}(a::IndexedObject{I,:N}) = IndexedObject{I,:C}(a.object,conj(a.α))
Base.conj{I}(a::IndexedObject{I,:C}) = IndexedObject{I,:N}(a.object,conj(a.α))

*{I,C}(a::IndexedObject{I,C},β::Number) = IndexedObject{I,C}(a.object,a.α*β)
*(β::Number,a::IndexedObject) = *(a,β)

indexed(object,::Indices{I}) = IndexedObject{I,:N}(A)
getindexed{I}(A::IndexedObject{I},::Indices{I}) = A.object
getindexed{I}(A::IndexedObject{I},::Indices{I}) = A.object
