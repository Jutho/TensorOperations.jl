#-------------------------------------------------------------------------------------------
# Operations
#-------------------------------------------------------------------------------------------

"""
    tensoradd!(C, pC, A, conjA, α=true, β=true [, backend])

Compute `C = β * C + α * permutedims(opA(A), pC)` without creating the intermediate
temporary. The operation `opA` acts as `identity` if `conjA` equals `:N` and as `conj`
if `conjA` equals `:C`. Optionally specify a backend implementation to use.

!!! warning
    The permutation needs to be trivial or `C` must not be aliased with `A`.

See also [`tensoradd`](@ref).
"""
function tensoradd! end
# insert default α and β arguments
function tensoradd!(C, pC::Index2Tuple, A, conjA::Symbol)
    return tensoradd!(C, pC, A, conjA, true, true)
end

"""
    tensortrace!(C, pC, A, pA, conjA, α=true, β=false [, backend])

Compute `C = β * C + α * permutedims(partialtrace(opA(A)), pC)` without creating the
intermediate temporary, where `A` is partially traced, such that indices in `pA[1]` are
contracted with indices in `pA[2]`, and the remaining indices are permuted according
to `pC`. The operation `opA` acts as `identity` if `conjA` equals `:N` and as `conj` if
`conjA` equals `:C`. Optionally specify a backend implementation to use.

!!! warning
    The object `C` must not be aliased with `A`.

See also [`tensortrace`](@ref).
"""
function tensortrace! end
# insert default α and β arguments
function tensortrace!(C, pC::Index2Tuple, A, pA::Index2Tuple, conjA::Symbol)
    return tensortrace!(C, pC, A, pA, conjA, true, false)
end

"""
    tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α=true, β=false [, backend])

Compute `C = β * C + α * permutedims(contract(opA(A), opB(B)), pC)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pC`. The operation `opA` acts as `identity` if `conjA`
equals `:N` and as `conj` if `conjA` equals `:C`; the operation `opB` is determined by
`conjB` analogously. Optionally specify a backend implementation to use.

!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See also [`tensorcontract`](@ref).
"""
function tensorcontract! end
# insert default α and β arguments
function tensorcontract!(C, pC::Index2Tuple,
                         A, pA::Index2Tuple, conjA::Symbol,
                         B, pB::Index2Tuple, conjB::Symbol)
    return tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, true, false)
end

"""
    tensorscalar(C)

Return the single element of a tensor-like object with zero indices or dimensions as
a value of the underlying scalar type.
"""
function tensorscalar end

#-------------------------------------------------------------------------------------------
# Allocations
#-------------------------------------------------------------------------------------------

"""
    tensorstructure(A, iA, conjA)

Obtain the information associated to indices `iA` of tensor `op(A)`, where `op` acts as
`identity` if `conjA` equals `:N` and as `conj` if `conjA` equals `:C`.
"""
function tensorstructure end

"""
    tensoradd_structure(pC, A, conjA)

Obtain the structure information of `C`, where `C` would be the output of
`tensoradd!(C, pC, A, conjA)`.
"""
function tensoradd_structure end

"""
    tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)

Obtain the structure information of `C`, where `C` would be the output of
`tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB)`.
"""
function tensorcontract_structure end

"""
    tensoradd_type(TC, pC, A, conjA)

Obtain `typeof(C)`, where `C` is the result of `tensoradd!(C, pC, A, conjA)`
with scalar type `TC`.
"""
function tensoradd_type end

"""
    tensorcontract_type(TC, pC, A, pA, conjA, B, pB, conjB)

Obtain `typeof(C)`, where `C` is the result of
`tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB)` with scalar type `TC`.
"""
function tensorcontract_type end

"""
    tensoralloc(ttype, structure, istemp=false, [backend::Backend])

Allocate memory for a tensor of type `ttype` and structure `structure`. The optional third
argument can be used to indicate that the result is used as a temporary tensor, for which
in some cases and with some backends (the optional fourth argument) a different allocation
strategy might be used.

See also [`tensoralloc_add`](@ref), [`tensoralloc_contract`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc end

"""
    tensorfree!(C, [backend::Backend])

Provide a hint that the allocated memory of `C` can be released.

See also [`tensoralloc`](@ref).
"""
tensorfree!(C) = nothing

#-------------------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------------------

"""
    tensorcost(A, i)
    
Compute the contraction cost associated with the `i`th index of a tensor, such that the
total cost of a pairwise contraction is found as the product of the costs of all contracted
indices and all uncontracted indices.
"""
function tensorcost end

"""
    checkcontractible(A, iA, conjA, B, iB, conjB, label)

Verify whether two tensors `opA(A)` and `opB(B)` are compatible for having their respective
index `iA` and `iB` contracted, and throws an error if not. The operation `opA` acts as
`identity` if `conjA` equals `:N` and as `conj` if `conjA` equals `:C`; the operation `opB`
is determined by `conjB` analogously.
"""
function checkcontractible end
