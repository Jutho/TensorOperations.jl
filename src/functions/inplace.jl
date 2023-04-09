# methods/inplace.jl
#
# Method-based access to tensor operations using inplace definitions.
tensorcopy!(C, A, IA, IC) = tensorcopy!(C, A, tuple(IA...), tuple(IC...))
tensoradd!(C, A, IA, IC, α, β) = tensoradd!(C, A, tuple(IA...), tuple(IC...), α, β)
tensortrace!(C, A, IA, IC, α, β) = tensortrace!(C, A, tuple(IA...), tuple(IC...), α, β)
function tensorcontract!(C, IC, A, IA, conjA, B, IB, conjB, α, β)
    return tensorcontract!(C, tuple(IC...), A, tuple(IA...), conjA, B, tuple(IB...), conjB,
                           α, β)
end
function tensorproduct!(α, A, IA, B, IB, β, C, IC)
    return tensorproduct!(α, A, tuple(IA...), B, tuple(IB...), β, C, tuple(IC...))
end

"""
    tensorcopy!(C, A, IA, IC)

Copies `A` into `C` by permuting the dimensions according to the pattern specified by
`IA` and `IC`. Both iterables should contain the same elements in a different order.
The result of this method is equivalent to `permutedims!(C, A, p)` where `p` is the
permutation such that `IC=IA[p]`. The implementation of tensorcopy! is however more
efficient on average, especially if `Threads.nthreads() > 1`.
"""
function tensorcopy!(C, A, IA::Tuple, IC::Tuple)
    return tensoradd!(C, A, IA, IC, one(scalartype(C)), zero(scalartype(C)))
end

"""
    tensoradd!(C, A, IA::Tuple, IC::Tuple, α, β)

Updates `C` to `β*C + α * tensorcopy(A,IA,IC)`, but without creating the temporary
permuted array.

See also: [`tensorcopy`](@ref)
"""
function tensoradd!(C, A, IA::Tuple, IC::Tuple, α, β)
    pC = add_indices(IA, IC)
    return tensoradd!(C, A, pC, :N, α, β)
end

"""
    tensortrace!(α, A, IA, β, C, IC)

Updates `C` to `β*C + α tensortrace(A,IA,IC)`, but without creating the temporary
traced array.

See also: [`tensortrace`](@ref)
"""
function tensortrace!(C, A, IA::Tuple, IC::Tuple, α, β)
    pC, cindA1, cindA2 = trace_indices(IA, IC)
    return tensortrace!(C, pC, A, (cindA1, cindA2), :N, α, β)
end

"""
    tensorcontract!(α, A, labelsA, conjA, B, labelsB, conjB, β, C, labelsC)

Replaces `C` with `β C + α A * B`, where some indices of array `A` are contracted with corresponding
indices in array `B` by assigning them identical labels in the iterables `labelsA` and `labelsB`.
The arguments `conjA` and `conjB` should be of type `Char` and indicate whether the data of
arrays `A` and `B`, respectively, need to be conjugated (value `'C'`) or not (value `'N'`).
Every label should appear exactly twice in the union of `labelsA`, `labelsB` and `labelsC`,
either in the intersection of `labelsA` and `labelsB` (for indices that need to be contracted)
or in the interaction of either `labelsA` or `labelsB` with `labelsC`, for indicating the order
in which the open indices should be match to the indices of the output array `C`.
"""
function tensorcontract!(C, IC::Tuple, A, IA::Tuple, conjA, B, IB::Tuple, conjB, α, β)
    conjA == 'N' || conjA == 'C' ||
        throw(ArgumentError("Value of conjA should be 'N' or 'C' instead of $conjA"))
    conjB == 'N' || conjB == 'C' ||
        throw(ArgumentError("Value of conjB should be 'N' or 'C' instead of $conjB"))
    CA = conjA == 'N' ? :N : :C
    CB = conjB == 'N' ? :N : :C

    pA, pB, pC = contract_indices(IA, IB, IC)
    return tensorcontract!(C, pC, A, pA, CA, B, pB, CB, α, β)
end

"""
    tensorproduct!(α, A, labelsA, B, labelsB, β, C, labelsC)

Replaces C with `β C + α A * B` without any indices being contracted.
"""
function tensorproduct!(C, IC::Tuple, A, IA::Tuple, B, IB::Tuple, α, β)
    isempty(intersect(IA, IB)) || throw(LabelError("not a valid tensor product"))
    return tensorcontract!(C, IC, A, IA, 'N', B, IB, 'N', α, β)
end
