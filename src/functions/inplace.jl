# methods/inplace.jl
#
# Method-based access to tensor operations using inplace definitions.

tensorcopy!(A, IA, C, IC) = tensoradd!(1, A, IA, 0, C, IC)

function tensoradd!(α, A, IA, β, C, IC)
    checkindices(A, IA)
    checkindices(C, IC)
    indCinA = add_indices(IA, IC)
    add!(α, A, Val{:N}, β, C, indCinA)
end

function tensortrace!(α, A, IA, β, C, IC)
    checkindices(A, IA)
    checkindices(C, IC)
    indCinA, cindA1, cindA2 = trace_indices(IA, IC)
    trace!(α, A, Val{:N}, β, C, indCinA, cindA1, cindA2)
    return C
end

function tensorcontract!(α, A, IA, conjA, B, IB, conjB, β, C, IC; method::Symbol = :BLAS)
    # Updates C as β*C+α*contract(A, B), whereby the contraction pattern
    # is specified by IA, IB and IC. The iterables IA(B, C)
    # should contain a unique label for every index of array A(B, C), such that
    # common I of A and B correspond to indices that will be contracted.
    # Common I between A and C or B and C indicate the position of the
    # uncontracted indices of A and B with respect to the indices of C, such
    # that the output array of the contraction can be added to C. Every label
    # should thus appear exactly twice in the union of IA, IB and
    # IC and the associated indices of the tensors should have identical
    # size.
    # Array A and/or B can be also conjugated by setting conjA and/or conjB
    # equal  to 'C' instead of 'N'.
    # The parametric argument method can be specified to choose between two
    # different contraction strategies:
    # -> method = :BLAS : permutes tensors (requires extra memory) and then
    #                   calls built-in (typically BLAS) multiplication
    # -> method = :native : memory-free native julia tensor contraction

    checkindices(A, IA)
    checkindices(B, IB)
    checkindices(C, IC)

    conjA == 'N' || conjA == 'C' || throw(ArgumentError("Value of conjA should be 'N' or 'C' instead of $conjA"))
    conjB == 'N' || conjB == 'C' || throw(ArgumentError("Value of conjB should be 'N' or 'C' instead of $conjB"))
    CA = conjA == 'N' ? :N : :C
    CB = conjB == 'N' ? :N : :C

    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)

    if method == :BLAS
        contract!(α, A, Val{CA}, B, Val{CB}, β, C, oindA, cindA, oindB, cindB, indCinoAB,Val{:BLAS})
    elseif method == :native
        contract!(α, A, Val{CA}, B, Val{CB}, β, C, oindA, cindA, oindB, cindB, indCinoAB,Val{:native})
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct!(α, A, IA, B, IB, β, C, IC)
    isempty(intersect(IA, IB)) || throw(LabelError("not a valid tensor product"))
    tensorcontract!(α, A, IA, 'N', B, IB, 'N', β, C, IC; method = :native)
end
