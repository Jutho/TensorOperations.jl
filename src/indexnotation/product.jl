# indexnotation/product.jl
#
# A wrapper to store the contraction (product) of two indexed objects and
# evaluate lazily, i.e. evaluate upon calling `deindexify`.

immutable ProductOfIndexedObjects{IA,IB,CA,CB,OA,OB,TA,TB} <: AbstractIndexedObject
    A::IndexedObject{IA,CA,OA,TA}
    B::IndexedObject{IB,CB,OB,TB}
end

Base.conj(P::ProductOfIndexedObjects) = ProductOfIndexedObjects(conj(P.A),conj(P.B))

Base.eltype(P::ProductOfIndexedObjects) = promote_type(eltype(P.A),eltype(P.B))

*(P::ProductOfIndexedObjects, β::Number) = ProductOfIndexedObjects(β*P.A, P.B)
*(β::Number, P::ProductOfIndexedObjects) = *(P,β)
-(P::ProductOfIndexedObjects) = *(-1, P)

@generated function indices{IA,IB,CA,CB,OA,OB,TA,TB}(::Type{ProductOfIndexedObjects{IA,IB,CA,CB,OA,OB,TA,TB}})
    J = unique2([IA...,IB...])
    J = tuple(J...)
    meta = Expr(:meta, :inline)
    Expr(:block, :meta, :($J))
end

@generated function *(A::AbstractIndexedObject,B::AbstractIndexedObject)
    IA = indices(A)
    IB = indices(B)
    IC = symdiff(IA,IB)
    oindA, cindA, oindB, cindB, = contract_indices(IA,IB,IC)
    meta = Expr(:meta,:inline)
    if A <: IndexedObject{IA}
        argA = :A
    else
        IA = IA[vcat(oindA,cindA)]
        IA = tuple(IA...)
        indA = :(Indices{$IA}())
        argA = :(indexify(deindexify(A,$indA),$indA))
    end
    if B <: IndexedObject{IB}
        argB = :B
    else
        IB = IB[vcat(cindB,oindB)]
        IB = tuple(IB...)
        indB = :(Indices{$IB}())
        argB = :(indexify(deindexify(B,$indB),$indB))
    end
    meta = Expr(:meta,:inline)
    Expr(:block,meta,:(ProductOfIndexedObjects($argA,$argB)))
end

@generated function deindexify{IA,IB,CA,CB,IC}(P::ProductOfIndexedObjects{IA,IB,CA,CB}, I::Indices{IC}, T::Type = eltype(P))
    meta = Expr(:meta, :inline)
    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)
    indCinAB = vcat(oindA,length(IA)+oindB)[indCinoAB]
    conjA = Val{CA}
    conjB = Val{CB}
    quote
        $meta
        deindexify!(similar_from_indices(T, $indCinAB, P.A.object, P.B.object, $conjA, $conjB), P, I)
    end
end

@generated function deindexify!{IA,IB,CA,CB,IC}(dst, P::ProductOfIndexedObjects{IA,IB,CA,CB}, ::Indices{IC}, β=0)
    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)
    conjA = Val{CA}
    conjB = Val{CB}
    meta = Expr(:meta,:inline)
    quote
        $meta
        contract!(P.A.α*P.B.α,P.A.object,$conjA,P.B.object,$conjB,β,dst,$oindA,$cindA,$oindB,$cindB,$indCinoAB)
    end
end
