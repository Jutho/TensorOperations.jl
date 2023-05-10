#===========================================================================================
    Operations
===========================================================================================#

"""
    tensoradd!(C, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(opA(A), pA)` without creating the intermediate
temporary.  The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
Note that the permutation needs to be trivial or `C` must not be aliased with `A`.
"""
function tensoradd! end

"""
    tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)

Implements `C = β * C + α * permutedims(contract(opA(A), opB(B)), pC)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pC`. The operation `opA` (`opB`) acts as `conj` if `conjA`
(`conjB`) equals `:C` or as the identity if `conjA` (`conjB`) equals `:N`.
Note that `C` must not be aliased with `A` or `B`.
"""
function tensorcontract! end

"""
    tensortrace!(C, pC, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(partialtrace(opA(A)), pC)` without creating the
intermediate temporary, where `A` is partially traced, such that indices in `pA[1]` are
contracted with indices in `pA[2]`, and the remaining indices are permuted according
to `pC`. The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
Note that `C` must not be aliased with `A`.
"""
function tensortrace! end

"""
    tensorscalar(C)

Returns the single element of a tensor-like object with zero indices or dimensions.
"""
function tensorscalar end

#===========================================================================================
    Allocations
===========================================================================================#

"""
    tensorstructure(A, iA, conjA)

Obtain the information associated to indices `iA` of tensor `op(A)`, where `op` acts as
`conj` when `conjA` is `:C`, or as the identity if `conjA` is `:N`.
"""
function tensorstructure end

"""
    tensoradd_structure(A, pA, conjA)

Obtain the structure information of `C`, where `C` would be the output of
`tensoradd!(C, A, pA, conjA)`.
"""
function tensoradd_structure(A, pA, conjA) end

"""
    tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)

Obtain the structure information of `C`, where `C` would be the output of
`tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB)`.
"""
function tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB) end

"""
    tensoradd_type(TC, A, pA, conjA)

Obtain `typeof(C)`, where `C` is the result of `tensoradd!(C, A, pA, conjA)` with scalartype
`TC`.
"""
function tensoradd_type(TC, A, pA, conjA) end

"""
    tensorcontract_type(TC, pC, A, pA, conjA, B, pB, conjB)

Obtain `typeof(C)`, where `C` is the result of
`tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB)` with scalartype `TC`.
"""
function tensorcontract_type(TC, pC, A, pA, conjA, B, pB, conjB) end

"""
    tensoralloc(ttype, structure, istemp=false)

Allocate memory for a tensor of type `ttype` and structure `structure`, optionally
distinguishing between temporary intermediate tensors.
"""
function tensoralloc end

"""
    tensorfree!(C)

Release the allocated memory of `C`.
"""
function tensorfree! end

#===========================================================================================
    Utility
===========================================================================================#

"""
    tensorcost(A, i)
    
Computes the contraction cost associated with the `i`th index of a tensor, such that the
total cost of a pairwise contraction is found as the product of the costs of all contracted
indices and all uncontracted indices.
"""
function tensorcost end

"""
    checkcontractible(A, iA, conjA, B, iB, conjB, label)
Verifies whether two tensors `A` and `B` are compatible for contraction, and throws an error
if not.
"""
function checkcontractible end
