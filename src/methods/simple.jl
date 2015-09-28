# methods/simple.jl
#
# Method-based access to tensor operations using simple definitions.

"""`tensorcopy(A, IA, IC=IA)`

Creates a copy of `A`, where the dimensions of `A` are assigned indices from the iterable `IA` and the indices of the copy are contained in `IC`. Both iterables should contain the same elements in a different order.

The result of this method is equivalent to `permutedims(A,p)`` where p is the permutation such that `IC=IA[p]`. The implementation of tensorcopy is however more efficient on average.
"""
function tensorcopy(A, IA, IC=IA)
    IA == IC && return copy(A)

    checkindices(A, IA)
    indCinA = add_indices(IA, IC)
    C = similar_from_indices(eltype(A), indCinA, A)
    add!(1, A, Val{:N}, 0, C, indCinA)
end

"""`tensoradd(A, IA, B, IB, IC=IA)`

Returns the result of adding arrays `A` and `B` where the iterabels `IA` and `IB` denote how the array data should be permuted in order to be added. More specifically, the result of this method is equivalent to

```
tensorcopy(A,IA,IC)+tensorcopy(B,IB,IC)
```

but without creating the temporary permuted arrays.
"""
function tensoradd(A, IA, B, IB, IC=IA)
    checkindices(A, IA)
    checkindices(B, IB)
    T = promote_type(eltype(A), eltype(B))
    if IA == IC
        C = similar_from_indices(T, 1:numind(A), A)
        copy!(C,A)
    else
        indCinA = add_indices(IA, IC)
        C = similar_from_indices(T, indCinA, A)
        add_native!(1, A, Val{:N}, 0, C, indCinA)
    end
    indCinB = add_indices(IB, IC)
    add!(1, B, Val{:N}, 1, C, indCinB)
end

"""`tensortrace(A, IA, IC = unique2(IA))`

Trace or contract pairs of indices of array `A`, by assigning them an identical indices in the iterable `IA`. The untraced indices, which are assigned a unique index, can be reordered according to the optional argument `IC`. The default value corresponds to the order in which they appear. Note that only pairs of indices can be contracted, so that every index in `IA` can appear only once (for an untraced index) or twice (for an index in a contracted pair).
"""
function tensortrace(A, IA, IC = unique2(IA))
    checkindices(A, IA)
    indCinA, cindA1, cindA2 = trace_indices(IA,IC)
    C = similar_from_indices(eltype(A), indCinA, A)
    trace!(1, A, Val{:N}, 0, C, indCinA, cindA1, cindA2)
end

function tensorcontract(A, IA, B, IB, IC = symdiff(IA,IB); method::Symbol = :BLAS)
    checkindices(A, IA)
    checkindices(B, IB)

    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)
    indCinAB = vcat(oindA,length(IA)+oindB)[indCinoAB]

    T = promote_type(eltype(A), eltype(B))
    C = similar_from_indices(T, indCinAB, A, B)

    if method == :BLAS
        contract!(1, A, Val{:N}, B, Val{:N}, 0, C, oindA, cindA, oindB, cindB, indCinoAB,Val{:BLAS})
    elseif method == :native
        contract!(1, A, Val{:N}, B, Val{:N}, 0, C, oindA, cindA, oindB, cindB, indCinoAB,Val{:native})
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct(A, IA, B, IB, IC = symdiff(IA,IB))
    isempty(intersect(IA, IB)) || throw(IndexError("not a valid tensor product"))
    tensorcontract(A, IA, B, IB, IC; method = :native)
end
