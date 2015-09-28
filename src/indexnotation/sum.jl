# indexnotation/sum.jl
#
# A wrapper to store the sum of a set of indexed objects and evaluate lazily,
# i.e. evaluate upon calling `deindexify`.

immutable SumOfIndexedObjects{Os<:Tuple{Vararg{AbstractIndexedObject}}} <: AbstractIndexedObject
    objects::Os
end

@inline _conjtuple(objects::Tuple{Any}) = tuple(conj(objects[1]))
@inline _conjtuple(objects::Tuple) = tuple(conj(objects[1]), _conjtuple(Base.tail(objects))...)
@inline _eltypetuple(objects::Tuple{Any}) = eltype(objects[1])
@inline _eltypetuple(objects::Tuple) = promote_type(eltype(objects[1]), _eltypetuple(Base.tail(objects)))
@inline _multuple(a, objects::Tuple{Any}) = tuple(a*objects[1])
@inline _multuple(a, objects::Tuple) = tuple(a*objects[1], _multuple(a, Base.tail(objects))...)

Base.conj(A::SumOfIndexedObjects) = SumOfIndexedObjects(_conjtuple(A.objects))

Base.eltype(A::SumOfIndexedObjects) = _eltypetuple(A.objects)

*(A::SumOfIndexedObjects, β::Number) = SumOfIndexedObjects(_multuple(β, A.objects))
*(β::Number, A::SumOfIndexedObjects) = SumOfIndexedObjects(_multuple(β, A.objects))
-(A::SumOfIndexedObjects) = *(-1, A)

indices{Os}(A::SumOfIndexedObjects{Os}) = indices(Os.parameters[1])

function +(A::SumOfIndexedObjects, B::SumOfIndexedObjects)
    add_indices(indices(A), indices(B)) # performs index check
    SumOfIndexedObjects(tuple(A.objects..., B.objects...))
end
function +(A::SumOfIndexedObjects, b::AbstractIndexedObject)
    add_indices(indices(A), indices(b)) # performs index check
    SumOfIndexedObjects(tuple(A.objects..., b))
end
function +(a::AbstractIndexedObject, B::SumOfIndexedObjects)
    add_indices(indices(a), indices(B)) # performs index check
    SumOfIndexedObjects(tuple(a, B.objects...))
end
function +(a::AbstractIndexedObject, b::AbstractIndexedObject)
    add_indices(indices(a), indices(b)) # performs index check
    SumOfIndexedObjects(tuple(a, b))
end

-(a::AbstractIndexedObject, b::AbstractIndexedObject) = +(a, -b)

@generated function deindexify{Os}(A::SumOfIndexedObjects{Os}, I::Indices)
    addex = Expr(:block, [:(deindexify!(dst, A.objects[$j], I, +1)) for j=2:length(Os.parameters)]...)
    quote
        dst = deindexify(A.objects[1], I, eltype(A))
        $addex
        return dst
    end
end

@generated function deindexify!{Os}(dst, A::SumOfIndexedObjects{Os}, I::Indices, β=0)
    addex = Expr(:block, [:(deindexify!(dst, A.objects[$j], I, +1)) for j=2:length(Os.parameters)]...)
    quote
        deindexify!(dst, A.objects[1], I, β)
        $addex
        return dst
    end
end
