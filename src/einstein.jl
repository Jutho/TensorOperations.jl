import Base: +, -, *

immutable Indices{I}
end

# Elementary object: IndexedObject parameterized by its indices, whether or not
# it has been conjugated, the type of the object it wraps, and the type of a
# possible scalar factor α
immutable IndexedObject{I,C,A,T} <: AbstractIndexedObject
    object::A
    α::T
end
Base.call{I,C,A,T}(::Type{IndexedObject{I,C}},object::A,α::T=1) = IndexedObject{I,C,A,T}(object,α)

Base.conj{I}(a::IndexedObject{I,:N}) = IndexedObject{I,:C}(a.object,conj(a.α))
Base.conj{I}(a::IndexedObject{I,:C}) = IndexedObject{I,:N}(a.object,conj(a.α))

*{I,C}(a::IndexedObject{I,C},β::Number) = IndexedObject{I,C}(a.object,a.α*β)
*(β::Number,a::IndexedObject) = *(a,β)


immutable ContractedNetwork{Is,As<:Tuple{Vararg{IndexedObject}}} <: AbstractIndexedObject
    objects::As
end


copy!{I,C}(dst::IndexedObject{I,C},src::IndexedObject{I,C}) = (copy!(dst.data,src.data);return dst)

@generated function copy!{I1,I2}(dst::IndexedObject{I1},src::IndexedObject{I2})
    indCinA = indexin(collect(I1),collect(I2))
    if length(I1) != length(I2) || !isperm(indCinA)
        throw(LabelError("non-matching labels: $I1 vs $I2")))
    end
    ex = :(add_native!(1, src.data, 0, dst.data, indCinA))
    Expr(:block,Expr(:meta,:inline),ex,:(return dst))
end

@generated function copy!{I1,I2,J}(dst::IndexedObject{I1},src::TracedIndexedObject{I2,J})
    labelsA = collect(J)
    labelsC = collect(I1)
    indCinA=indexin(labelsC,labelsA)

    clabels=unique(setdiff(labelsA,labelsC))
    cindA1=Array(Int,length(clabels))
    cindA2=Array(Int,length(clabels))
    for i=1:length(clabels)
        cindA1[i] = findfirst(labelsA,clabels[i])
        cindA2[i] = findnext(labelsA,clabels[i],cindA1[i]+1)
    end
    pA = vcat(indCinA, cindA1, cindA2)
    isperm(pA) || throw(LabelError("invalid trace specification: $J -> $I1"))
    ex = :(trace_native!(1, A, 0, C, $indCinA, $cindA1, $cindA2))




    indCinA = indexin(collect(I1),collect(I2))
    tracelabels = set
    if length(I1) != length(I2) || !isperm(indCinA)
        throw(LabelError("non-matching labels: $I1 vs $I2")))
    end
    ex = :(add_native!(1, src.data, 0, dst.data, indCinA))
    Expr(:block,Expr(:meta,:inline),ex,:(return dst))
end



@generated function Base.getindex{J}(A::StridedArray,::Indices{J})
    length(J) == ndims(A) || throw(LabelError("wrong number of indices: $I"))
    I = tuple(unique2(J)...)
    if I==J
        ex = :( IndexedObject{$I,:N}(A) )
    else
        ex = :( IndexedObject{$I,:N}(tensortrace(A) )
    end
    Expr(:block,Expr(:meta,:inline),ex)
end

Base.conj{A,I}(a::IndexedObject{A,I,:N}) = IndexedObject{A,I,:C}(a.array)
Base.conj{A,I}(a::IndexedObject{A,I,:C}) = IndexedObject{A,I,:N}(a.array)
