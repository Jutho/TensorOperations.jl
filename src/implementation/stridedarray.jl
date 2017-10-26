# implementation/stridedarray.jl
#
# High-level implementation of tensor operations for StridedArray from Julia
# Base Library. Checks dimensions and converts to StridedData before passing
# to low-level (recursive) function.
add!(α, A::StridedArray, CA::Type{<:Val}, β, C::StridedArray, p1::IndexTuple, p2::IndexTuple) = add!(α, A, CA, β, C, (p1...,p2...))

"""
    add!(α, A, conjA, β, C, indCinA)

Implements `C = β*C+α*permute(op(A))` where `A` is permuted according to `indCinA`
and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The
indexable collection `indCinA` contains as nth entry the dimension of `A` associated
with the nth dimension of `C`.
"""
function add!(α, A::StridedArray, ::Type{Val{CA}}, β, C::StridedArray, indCinA) where CA
    for i = 1:ndims(C)
        size(A, indCinA[i]) == size(C, i) || throw(DimensionMismatch("$(size(A)), $(size(C)), $(indCinA)"))
    end

    dims, stridesA, stridesC, minstrides = add_strides(size(C), _permute(strides(A), indCinA), strides(C))
    dataA = StridedData(A, stridesA, Val{CA})
    offsetA = 0
    dataC = StridedData(C, stridesC)
    offsetC = 0

    if α == 0
        β == 1 || _scale!(dataC,β,dims)
    elseif α == 1 && β == 0
        add_rec!(_one, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif α == 1 && β == 1
        add_rec!(_one, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 0
        add_rec!(α, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 1
        add_rec!(α, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    else
        add_rec!(α, dataA, β, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end

trace!(α, A::StridedArray, CA::Type{<:Val}, β, C::StridedArray, p1, p2, cindA1, cindA2) = trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)

"""
    trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)

Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced,
according to `indCinA`, `cindA1` and `cindA2`, and `op` is `conj` if `conjA=Val{:C}`
or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains
as nth entry the dimension of `A` associated with the nth dimension of `C`. The
partial trace is performed by contracting dimension `cindA1[i]` of `A` with dimension
`cindA2[i]` of `A` for all `i in 1:length(cindA1)`.
"""
function trace!(α, A::StridedArray, ::Type{Val{CA}}, β, C::StridedArray, indCinA, cindA1, cindA2) where CA
    NC = ndims(C)
    NA = ndims(A)

    for i = 1:NC
        size(A,indCinA[i]) == size(C,i) || throw(DimensionMismatch(""))
    end
    map(i->size(A,i), cindA1) == map(i->size(A,i), cindA2) || throw(DimensionMismatch(""))

    pA = (indCinA..., cindA1..., cindA2...)
    dims, stridesA, stridesC, minstrides = trace_strides(_permute(size(A), pA), _permute(strides(A), pA), strides(C))
    dataA = StridedData(A, stridesA, Val{CA})
    offsetA = 0
    dataC = StridedData(C, stridesC)
    offsetC = 0

    if α == 0
        β == 1 || _scale!(dataC, β, dims)
    elseif α == 1 && β == 0
        trace_rec!(_one, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif α == 1 && β == 1
        trace_rec!(_one, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 0
        trace_rec!(α, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif β == 1
        trace_rec!(α, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    else
        trace_rec!(α, dataA, β, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end


contract!(α, A::StridedArray, CA::Type{<:Val}, B::StridedArray, CB::Type{<:Val}, β, C::StridedArray, oindA, cindA, oindB, cindB, p1, p2, method::Type{<:Val} = Val{:BLAS}) =
    contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1..., p2...), method)
"""
    contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, [method])

Implements `C = β*C+α*contract(op(A),op(B))` where `A` and `B` are contracted according
to `oindA`, `cindA`, `oindB`, `cindB` and `indCinoAB`. The operation `op` acts as
`conj` if `conjA` or `conjB` equal `Val{:C}` or as the identity map if `conjA` (`conjB`)
equal `Val{:N}`. The dimension `cindA[i]` of `A` is contracted with dimension `cindB[i]`
of `B`. The `n`th dimension of C is associated with an uncontracted (open) dimension
of `A` or `B` according to `indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA]`
with `NoA=length(oindA)` the number of open dimensions of `A`.

The optional argument `method` specifies whether the contraction is performed using
BLAS matrix multiplication by specifying `Val{:BLAS}` (default), or using a native
algorithm by specifying `Val{:native}`. The native algorithm does not copy the data
but is typically slower.
"""
function contract!(α, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, β, C::StridedArray{TC}, oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{:BLAS}}=Val{:BLAS}) where {CA,CB,TC<:Base.LinAlg.BlasFloat}
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    TA = eltype(A)
    TB = eltype(B)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = map(i->dimA[i], cindA)
    cdimsB = map(i->dimB[i], cindB)
    odimsA = map(i->dimA[i], oindA)
    odimsB = map(i->dimB[i], oindB)
    odimsAB = tuple(odimsA..., odimsB...)

    cdimsA == cdimsB || throw(DimensionMismatch())
    cdims = cdimsA

    for i = 1:length(indCinoAB)
        dimC[i] == odimsAB[indCinoAB[i]] || throw(DimensionMismatch())
    end

    olengthA = prod(odimsA)
    olengthB = prod(odimsB)
    clength = prod(cdims)

    # permute A
    if CA == :C
        conjA = 'C'
        pA = (cindA..., oindA...)
        if isa(A, Array{TC}) && pA == (1:NA...)
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}((cdims..., odimsA...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add!(1, A, Val{:N}, 0, Apermuted, pA)
            Amat = reshape(Apermuted, (clength, olengthA))
        end
    else
        conjA = 'N'
        pA = (oindA..., cindA...)
        if isa(A, Array{TC}) && pA == (1:NA...)
            Amat = reshape(A, (olengthA, clength))
        elseif isa(A, Array{TC}) && (cindA..., oindA...) == (1:NA...)
            conjA = 'T'
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}((odimsA..., cdims...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add!(1, A, Val{:N}, 0, Apermuted, pA)
            Amat = reshape(Apermuted, (olengthA, clength))
        end
    end

    # permute B
    if CB == :C
        conjB = 'C'
        pB = (oindB..., cindB...)
        if isa(B, Array{TC}) && pB == (1:NB...)
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}((odimsB..., cdims...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add!(1, B, Val{:N}, 0, Bpermuted, pB)
            Bmat = reshape(Bpermuted, (olengthB, clength))
        end
    else
        conjB = 'N'
        pB = (cindB..., oindB...)
        if  isa(B, Array{TC}) && pB == (1:NB...)
            Bmat = reshape(B, (clength, olengthB))
        elseif isa(B, Array{TC}) && (oindB..., cindB...) == (1:NB...)
            conjB = 'T'
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}((cdims..., odimsB...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add!(1, B, Val{:N}, 0, Bpermuted, pB)
            Bmat = reshape(Bpermuted, (clength, olengthB))
        end
    end

    # calculate C
    if isa(C, Array) && indCinoAB == (1:NC...)
        Cmat = reshape(C, (olengthA, olengthB))
        BLAS.gemm!(conjA, conjB, TC(α), Amat, Bmat, TC(β), Cmat)
    else
        Cmat = Array{TC}(olengthA, olengthB)
        BLAS.gemm!(conjA, conjB, TC(1), Amat, Bmat, TC(0), Cmat)
        add!(α, reshape(Cmat, (odimsA..., odimsB...)), Val{:N}, β, C, indCinoAB)
    end
    return C
end

function contract!(α, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, β, C::StridedArray, oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{:native}}=Val{:native}) where {CA,CB}
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = map(i->dimA[i], cindA)
    cdimsB = map(i->dimB[i], cindB)
    odimsA = map(i->dimA[i], oindA)
    odimsB = map(i->dimB[i], oindB)
    odimsAB = tuple(odimsA..., odimsB...)

    cdimsA == cdimsB || throw(DimensionMismatch())

    for i = 1:length(indCinoAB)
        dimC[i] == odimsAB[indCinoAB[i]] || throw(DimensionMismatch())
    end

    # Perform contraction
    pA = (oindA..., cindA...)
    pB = (oindB..., cindB...)
    sA = _permute(strides(A), pA)
    sB = _permute(strides(B), pB)
    sC = _permute(strides(C), tinvperm(indCinoAB))

    dimsA = _permute(size(A), pA)
    dimsB = _permute(size(B), pB)

    dims, stridesA, stridesB, stridesC, minstrides = contract_strides(dimsA, dimsB, sA, sB, sC)
    offsetA = offsetB = offsetC = 0
    dataA = StridedData(A, stridesA, Val{CA})
    dataB = StridedData(B, stridesB, Val{CB})
    dataC = StridedData(C, stridesC)

    # contract via recursive divide and conquer
    if α == 0
        β == 1 || _scale!(dataC, β, dims)
    elseif α == 1 && β == 0
        contract_rec!(_one, dataA, dataB, _zero, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif α == 1 && β == 1
        contract_rec!(_one, dataA, dataB, _one, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif β == 0
        contract_rec!(α, dataA, dataB, _zero, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif β == 1
        contract_rec!(α, dataA, dataB, _one, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    else
        contract_rec!(α, dataA, dataB, β, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    end
    return C
end
