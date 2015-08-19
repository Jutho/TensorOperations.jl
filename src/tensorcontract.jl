# tensorcontract.jl
#
# Method for contracting two tensors and adding the result
# to a third tensor, according to the specified labels.
# Method for computing the tensorproduct of two tensors as special
# case.

# Extract index information
#---------------------------
function contract_indices(labelsA, labelsB, labelsC)
    # Compute contraction indices and check for valid permutation
    NA = length(labelsA)
    NB = length(labelsB)
    NC = length(labelsC)

    NA == length(unique(labelsA)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsA"))
    NB == length(unique(labelsB)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsB"))
    NC == length(unique(labelsC)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsC"))

    clabels = intersect(labelsA, labelsB)
    numcontract = length(clabels)
    olabelsA = intersect(labelsC, labelsA)
    numopenA = length(olabelsA)
    olabelsB = intersect(labelsC, labelsB)
    numopenB = length(olabelsB)

    if numcontract+numopenA != NA || numcontract+numopenB != NB || numopenA+numopenB != NC
        throw(LabelError("invalid contraction pattern"))
    end

    cindA = indexin(clabels, collect(labelsA))
    oindA = indexin(olabelsA, collect(labelsA))
    cindB = indexin(clabels, collect(labelsB))
    oindB = indexin(olabelsB, collect(labelsB))
    indCinoAB = indexin(labelsC, vcat(olabelsA, olabelsB))

    if !isperm(vcat(oindA, cindA)) || !isperm(vcat(oindB, cindB)) || !isperm(indCinoAB)
        throw(LabelError("invalid contraction pattern: $labelsA and $labelsB to $labelsC"))
    end

    return oindA, cindA, oindB, cindB, indCinoAB
end

# Simple method
#---------------
function tensorcontract(A, labelsA, B, labelsB, labelsC = symdiff(labelsA, labelsB); method::Symbol = :BLAS)
    checklabellength(A, labelsA)
    checklabellength(B, labelsB)

    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(labelsA, labelsB, labelsC)
    indCinAB = vcat(oindA,length(labelsA)+oindB)[indCinoAB]

    T = promote_type(eltype(A), eltype(B))
    C = similar_from_indices(T, indCinAB, A, B)

    if method == :BLAS
        contract_blas!(1, A, Val{:N}, B, Val{:N}, 0, C, oindA, cindA, oindB, cindB, indCinoAB)
    elseif method == :native
        contract_native!(1, A, Val{:N}, B, Val{:N}, 0, C, oindA, cindA, oindB, cindB, indCinoAB)
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct(A, labelsA, B, labelsB, labelsC = vcat(labelsA, labelsB))
    isempty(intersect(labelsA, labelsB)) || throw(LabelError("not a valid tensor product"))
    tensorcontract(A, labelsA, B, labelsB, labelsC; method = :native)
end

# In-place method
#-----------------
function tensorcontract!(alpha, A, labelsA, conjA, B, labelsB, conjB, beta, C, labelsC; method::Symbol = :BLAS)
    # Updates C as beta*C+alpha*contract(A, B), whereby the contraction pattern
    # is specified by labelsA, labelsB and labelsC. The iterables labelsA(B, C)
    # should contain a unique label for every index of array A(B, C), such that
    # common labels of A and B correspond to indices that will be contracted.
    # Common labels between A and C or B and C indicate the position of the
    # uncontracted indices of A and B with respect to the indices of C, such
    # that the output array of the contraction can be added to C. Every label
    # should thus appear exactly twice in the union of labelsA, labelsB and
    # labelsC and the associated indices of the tensors should have identical
    # size.
    # Array A and/or B can be also conjugated by setting conjA and/or conjB
    # equal  to 'C' instead of 'N'.
    # The parametric argument method can be specified to choose between two
    # different contraction strategies:
    # -> method = :BLAS : permutes tensors (requires extra memory) and then
    #                   calls built-in (typically BLAS) multiplication
    # -> method = :native : memory-free native julia tensor contraction

    checklabellength(A, labelsA)
    checklabellength(B, labelsB)
    checklabellength(C, labelsC)

    conjA == 'N' || conjA == 'C' || throw(ArgumentError("Value of conjA should be 'N' or 'C' instead of $conjA"))
    conjB == 'N' || conjB == 'C' || throw(ArgumentError("Value of conjB should be 'N' or 'C' instead of $conjB"))
    CA = conjA == 'N' ? :N : :C
    CB = conjB == 'N' ? :N : :C

    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(labelsA, labelsB, labelsC)

    if method == :BLAS
        contract_blas!(alpha, A, Val{CA}, B, Val{CB}, beta, C, oindA, cindA, oindB, cindB, indCinoAB)
    elseif method == :native
        contract_native!(alpha, A, Val{CA}, B, Val{CB}, beta, C, oindA, cindA, oindB, cindB, indCinoAB)
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct!(alpha, A, labelsA, B, labelsB, beta, C, labelsC)
    isempty(intersect(labelsA, labelsB)) || throw(LabelError("not a valid tensor product"))
    tensorcontract!(alpha, A, labelsA, 'N', B, labelsB, 'N', beta, C, labelsC; method = :native)
end

# Implementation methods
#------------------------
# High level: can be extended for other types of arrays or tensors
function contract_blas!{CA,CB}(alpha, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, beta, C::StridedArray, oindA, cindA, oindB, cindB, indCinoAB)
    # The :BLAS method specification permutes A and B such that indopen and
    # indcontract are grouped, reshape them to matrices with all indopen on one
    # side and all indcontract on the other. Compute the data for C from
    # multiplying these matrices. Permute again to bring indices in requested
    # order.

    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = dimA[cindA]
    cdimsB = dimB[cindB]
    odimsA = dimA[oindA]
    odimsB = dimB[oindB]
    odimsAB = tuple(odimsA..., odimsB...)

    for i = 1:length(cdimsA)
        cdimsA[i] == cdimsB[i] || throw(DimensionMismatch())
    end
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
        pA = vcat(cindA, oindA)
        if isa(A, Array{TC}) && pA == collect(1:NA)
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}(tuple(cdims..., odimsA...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add_native!(1, A, Val{:N}, 0, Apermuted, pA)
            Amat = reshape(Apermuted, (clength, olengthA))
        end
    else
        conjA = 'N'
        pA = vcat(oindA, cindA)
        if isa(A, Array{TC}) && pA == collect(1:NA)
            Amat = reshape(A, (olengthA, clength))
        elseif isa(A, Array{TC}) && vcat(cindA, oindA) == collect(1:NA)
            conjA = 'T'
            Amat = reshape(A, (clength, olengthA))
        else
            Apermuted = Array{TC}(tuple(odimsA..., cdims...))
            # tensorcopy!(A, 1:NA, Apermuted, pA)
            add_native!(1, A, Val{:N}, 0, Apermuted, pA)
            Amat = reshape(Apermuted, (olengthA, clength))
        end
    end

    # permute B
    if CB == :C
        conjB = 'C'
        pB = vcat(oindB, cindB)
        if isa(B, Array{TC}) && pB == collect(1:NB)
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}(tuple(odimsB..., cdims...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add_native!(1, B, Val{:N}, 0, Bpermuted, pB)
            Bmat = reshape(Bpermuted, (olengthB, clength))
        end
    else
        conjB = 'N'
        pB = vcat(cindB, oindB)
        if  isa(B, Array{TC}) && pB == collect(1:NB)
            Bmat = reshape(B, (clength, olengthB))
        elseif isa(B, Array{TC}) && vcat(oindB, cindB) == collect(1:NB)
            conjB = 'T'
            Bmat = reshape(B, (olengthB, clength))
        else
            Bpermuted = Array{TC}(tuple(cdims..., odimsB...))
            # tensorcopy!(B, 1:NB, Bpermuted, pB)
            add_native!(1, B, Val{:N}, 0, Bpermuted, pB)
            Bmat = reshape(Bpermuted, (clength, olengthB))
        end
    end

    # calculate C
    if isa(C, Array) && indCinoAB == collect(1:NC)
        Cmat = reshape(C, (olengthA, olengthB))
        BLAS.gemm!(conjA, conjB, TC(alpha), Amat, Bmat, TC(beta), Cmat)
    else
        Cmat = Array{TC}(olengthA, olengthB)
        BLAS.gemm!(conjA, conjB, TC(1), Amat, Bmat, TC(0), Cmat)
        # tensoradd!(alpha, reshape(Cmat, tuple(odimsA..., odimsB...)), pC, beta, C, 1:NC)
        add_native!(alpha, reshape(Cmat, tuple(odimsA..., odimsB...)), Val{:N}, beta, C, indCinoAB)
    end
    return C
end

# High level: can be extended for other types of arrays or tensors
function contract_native!{CA,CB}(alpha, A::StridedArray, ::Type{Val{CA}}, B::StridedArray, ::Type{Val{CB}}, beta, C::StridedArray, oindA, cindA, oindB, cindB, indCinoAB)
    # native contraction method using divide and conquer

    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = dimA[cindA]
    cdimsB = dimB[cindB]
    odimsA = dimA[oindA]
    odimsB = dimB[oindB]
    odimsAB = tuple(odimsA..., odimsB...)

    # Perform contraction
    pA = vcat(oindA, cindA)
    pB = vcat(oindB, cindB)
    sA = _permute(_strides(A), pA)
    sB = _permute(_strides(B), pB)
    sC = _permute(_strides(C), invperm(indCinoAB))

    dimsA = _permute(size(A), pA)
    dimsB = _permute(size(B), pB)

    dims, stridesA, stridesB, stridesC, minstrides = contract_strides(dimsA, dimsB, sA, sB, sC)
    offsetA = offsetB = offsetC = 0
    dataA = StridedData(A, stridesA, Val{CA})
    dataB = StridedData(B, stridesB, Val{CB})
    dataC = StridedData(C, stridesC)

    # contract via recursive divide and conquer
    if alpha == 0
        beta == 1 || _scale!(dataC, beta, dims)
    elseif alpha == 1 && beta == 0
        contract_rec!(_one, dataA, dataB, _zero, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif alpha == 1 && beta == 1
        contract_rec!(_one, dataA, dataB, _one, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif beta == 0
        contract_rec!(alpha, dataA, dataB, _zero, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    elseif beta == 1
        contract_rec!(alpha, dataA, dataB, _one, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    else
        contract_rec!(alpha, dataA, dataB, beta, dataC, dims, offsetA, offsetB, offsetC, minstrides)
    end
    return C
end

# Recursive divide and conquer approach:
@generated function contract_rec!{N}(alpha, A::StridedData{N}, B::StridedData{N}, beta, C::StridedData{N},
    dims::NTuple{N, Int}, offsetA::Int, offsetB::Int, offsetC::Int, minstrides::NTuple{N, Int})

    quote
        odimsA = _filterdims(_filterdims(dims, A), C)
        odimsB = _filterdims(_filterdims(dims, B), C)
        cdims = _filterdims(_filterdims(dims, A), B)
        oAlength = prod(odimsA)
        oBlength = prod(odimsB)
        clength = prod(cdims)

        if oAlength*oBlength + clength*(oAlength+oBlength) <= BASELENGTH
            contract_micro!(alpha, A, B, beta, C, dims, offsetA, offsetB, offsetC)
        else
            if clength > oAlength && clength > oBlength
                dmax = _indmax(_memjumps(cdims, minstrides))
            elseif oAlength > oBlength
                dmax = _indmax(_memjumps(odimsA, minstrides))
            else
                dmax = _indmax(_memjumps(odimsB, minstrides))
            end
            @dividebody $N dmax dims offsetA A offsetB B offsetC C begin
                    contract_rec!(alpha, A, B, beta, C, dims, offsetA, offsetB, offsetC, minstrides)
                end begin
                if C.strides[dmax] == 0 # dmax is contraction dimension: beta -> 1
                    contract_rec!(alpha, A, B, _one, C, dims, offsetA, offsetB, offsetC, minstrides)
                else
                    contract_rec!(alpha, A, B, beta, C, dims, offsetA, offsetB, offsetC, minstrides)
                end
            end
        end
        return C
    end
end

# Micro kernel at end of recursion
@generated function contract_micro!{N}(alpha, A::StridedData{N}, B::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA, offsetB, offsetC)
    quote
        _scale!(C, beta, dims, offsetC)
        startA = A.start+offsetA
        stridesA = A.strides
        startB = B.start+offsetB
        stridesB = B.strides
        startC = C.start+offsetC
        stridesC = C.strides

        @stridedloops($N, dims, indA, startA, stridesA, indB, startB, stridesB, indC, startC, stridesC, @inbounds C[indC] = axpby(alpha, A[indA]*B[indB], _one, C[indC]))
        return C
    end
end

# Stride calculation
#--------------------
@generated function contract_strides{NA, NB, NC}(dimsA::NTuple{NA, Int}, dimsB::NTuple{NB, Int},
    stridesA::NTuple{NA, Int}, stridesB::NTuple{NB, Int}, stridesC::NTuple{NC, Int})
    meta = Expr(:meta, :inline)
    cN = div(NA+NB-NC, 2)
    oNA = NA - cN
    oNB = NB - cN

    dimsex = Expr(:tuple, [:(dimsA[$d]) for d = 1:oNA]..., [:(dimsB[$d]) for d = 1:oNB]..., [:(dimsA[$(oNA+d)]) for d = 1:cN]...)

    stridesAex = Expr(:tuple, [:(stridesA[$d]) for d = 1:oNA]..., [0 for d = 1:oNB]..., [:(stridesA[$(oNA+d)]) for d = 1:cN]...)
    stridesBex = Expr(:tuple, [0 for d = 1:oNA]..., [:(stridesB[$d]) for d = 1:oNB]..., [:(stridesB[$(oNB+d)]) for d = 1:cN]...)
    stridesCex = Expr(:tuple, [:(stridesC[$d]) for d = 1:(oNA+oNB)]..., [0 for d = 1:cN]...)

    minstridesex = Expr(:tuple, [:(min(stridesA[$d], stridesC[$d])) for d = 1:oNA]...,
    [:(min(stridesB[$d], stridesC[$(oNA+d)])) for d = 1:oNB]...,
    [:(min(stridesA[$(oNA+d)], stridesB[$(oNB+d)])) for d = 1:cN]...)
    quote
        $meta
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute($dimsex, p)
        stridesA = _permute($stridesAex, p)
        stridesB = _permute($stridesBex, p)
        stridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesB, stridesC, minstrides
    end
end
