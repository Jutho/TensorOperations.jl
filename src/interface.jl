#-------------------------------------------------------------------------------------------
# Operations
#-------------------------------------------------------------------------------------------

"""
    tensoradd!(C, A, pA, conjA, α=1, β=1 [, backend])

Compute `C = β * C + α * permutedims(opA(A), pA)` without creating the intermediate
temporary. The operation `opA` acts as `identity` if `conjA` equals `:N` and as `conj`
if `conjA` equals `:C`. Optionally specify a backend implementation to use.

!!! warning
    The permutation needs to be trivial or `C` must not be aliased with `A`.

See also [`tensoradd`](@ref).
"""
function tensoradd! end
# insert default α and β arguments
function tensoradd!(C, A, pA::Index2Tuple, conjA::Bool)
    return tensoradd!(C, A, pA, conjA, One(), One())
end
# insert default backend
function tensoradd!(C, A, pA::Index2Tuple, conjA::Bool, α::Number, β::Number)
    return tensoradd!(C, A, pA, conjA, α, β, select_backend(tensoradd!, C, A))
end

"""
    tensortrace!(C, A, p, q, conjA, α=1, β=0 [, backend])

Compute `C = β * C + α * permutedims(opA(A), (p, q))` without creating the
intermediate temporary, where `A` is partially traced, such that indices in `q[1]` are
contracted with indices in `q[2]`, and the other indices are permuted according
to `p`. The operation `opA` acts as `identity` if `conjA` equals `:N` and as `conj` if
`conjA` equals `:C`. Optionally specify a backend implementation to use.

!!! warning
    The object `C` must not be aliased with `A`.

See also [`tensortrace`](@ref).
"""
function tensortrace! end
# insert default α and β arguments
function tensortrace!(C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool)
    return tensortrace!(C, A, p, q, conjA, One(), Zero())
end
# insert default backend
function tensortrace!(C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number,
                      β::Number)
    return tensortrace!(C, A, p, q, conjA, α, β, select_backend(tensortrace!, C, A))
end

"""
    tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α=1, β=0 [, backend])

Compute `C = β * C + α * permutedims(contract(opA(A), opB(B)), pAB)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pAB`. The operation `opA` acts as `identity` if `conjA`
equals `:N` and as `conj` if `conjA` equals `:C`; the operation `opB` is determined by
`conjB` analogously. Optionally specify a backend implementation to use.

!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See also [`tensorcontract`](@ref).
"""
function tensorcontract! end
# insert default α and β arguments
function tensorcontract!(C,
                         A, pA::Index2Tuple, conjA::Bool,
                         B, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple)
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, One(), Zero())
end
function tensorcontract!(C,
                         A, pA::Index2Tuple, conjA::Bool,
                         B, pB::Index2Tuple, conjB::Bool,
                         pAB::Index2Tuple,
                         α::Number, β::Number)
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
                           select_backend(tensorcontract!, C, A, B))
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
    tensoradd_structure(A, pA, conjA)

Obtain the structure information of `C`, where `C` would be the output of
`tensoradd!(C, A, pA, conjA)`.
"""
function tensoradd_structure end

"""
    tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)

Obtain the structure information of `C`, where `C` would be the output of
`tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)`.
"""
function tensorcontract_structure end

"""
    tensoradd_type(TC, A, pA, conjA)

Obtain the type information of `C`, where `C` would be the output of
`tensoradd!(C, A, pA, conjA)` with scalartype `TC`.
"""
function tensoradd_type end

"""
    tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)

Obtain the type information of `C`, where `C` would be the output of
`tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)` with scalar type `TC`.
"""
function tensorcontract_type end

"""
    tensoralloc(ttype, structure, istemp=false, [backend::AbstractBackend])

Allocate memory for a tensor of type `ttype` and structure `structure`. The optional third
argument can be used to indicate that the result is used as a temporary tensor, for which
in some cases and with some backends (the optional fourth argument) a different allocation
strategy might be used.

See also [`tensoralloc_add`](@ref), [`tensoralloc_contract`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc end

"""
    tensorfree!(C, [backend::AbstractBackend])

Provide a hint that the allocated memory of `C` can be released.

See also [`tensoralloc`](@ref).
"""
function tensorfree! end

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
