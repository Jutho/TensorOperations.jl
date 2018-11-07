# For `AbstractArray`, we do not differentiate between left and right indices:

"""
    checked_similar_from_indices(C, T, indleft::IndexTuple, indright::IndexTuple, A, conjA = :N)

Returns an object similar to `A` which has an `eltype` given by `T` and whose left indices
correspond to the indices `indleft` from `op(A)`, and its right indices correspond to the
indices `indright` from `op(A)`, where `op` is `conj` if `conjA=:C` or does nothing if `conjA=:N`
(default). Here, `C` is a potential candidate for the similar object. If `C === nothing`, or its
its `eltype` or shape does not match, a new object is allocated and returned. Otherwise, `C`
is returned.
"""
checked_similar_from_indices(C, T::Type, p1::IndexTuple, p2::IndexTuple, A::AbstractArray,
    CA::Symbol = :N) = checked_similar_from_indices(C, T, (p1..., p2...), A, CA)

"""
    checked_similar_from_indices(C, T, indoA, indoB, indleft, indright, A, B, conjA = :N, conjB= :N)

Returns an object similar to `A` which has an `eltype` given by `T` and dimensions/sizes
corresponding to a selection of those of `op(A)` and `op(B)` concatenated, where the
selection is specified by `indices` (which contains integers between `1` and
    `numind(A)+numind(B)` and `op` is `conj` if `conjA` or `conjB` equal `:C`
    or does nothing if `conjA` or `conjB` equal `:N` (default).
"""
checked_similar_from_indices(C, T::Type, poA::IndexTuple, poB::IndexTuple, p1::IndexTuple,
    p2::IndexTuple, A::AbstractArray, B::AbstractArray, CA::Symbol = :N, CB::Symbol = :N) =
    checked_similar_from_indices(C, T, poA, poB, (p1..., p2...), A, B, CA, CB)

"""
    scalar(C)

Returns the single element of a tensor-like object with zero dimensions, i.e. if `numind(C)==0`.
"""
function scalar end

"""
    add!(α, A, conjA, β, C, indCinA)

Implements `C = β*C+α*permute(op(A))` where `A` is permuted according to `indCinA`
and `op` is `conj` if `conjA=:C` or the identity map if `conjA=:N`. The
indexable collection `indCinA` contains as `n`th entry the dimension of `A` associated
with the `n`th dimension of `C`.
"""
add!(α, A::AbstractArray, CA::Symbol, β, C::AbstractArray, p1, p2) = add!(α, A, CA, β, C, (p1...,p2...))

"""
    trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)

Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced,
according to `indCinA`, `cindA1` and `cindA2`, and `op` is `conj` if `conjA=:C`
or the identity map if `conjA=:N`. The indexable collection `indCinA` contains
as nth entry the dimension of `A` associated with the nth dimension of `C`. The
partial trace is performed by contracting dimension `cindA1[i]` of `A` with dimension
`cindA2[i]` of `A` for all `i in 1:length(cindA1)`.
"""
trace!(α, A::AbstractArray, CA::Symbol, β, C::AbstractArray, p1, p2, cindA1, cindA2) =
    trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)

"""
    contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB)

Implements `C = β*C+α*contract(op(A),op(B))` where `A` and `B` are contracted according
to `oindA`, `cindA`, `oindB`, `cindB` and `indCinoAB`. The operation `op` acts as
`conj` if `conjA` or `conjB` equal `Val(:C)` or as the identity map if `conjA` (`conjB`)
equal `Val(:N)`. The dimension `cindA[i]` of `A` is contracted with dimension `cindB[i]`
of `B`. The `n`th dimension of C is associated with an uncontracted (open) dimension
of `A` or `B` according to `indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA]`
with `NoA=length(oindA)` the number of open dimensions of `A`.

The optional argument `method` specifies whether the contraction is performed using
BLAS matrix multiplication by specifying `Val(:BLAS)` (default), or using a native
algorithm by specifying `Val(:native)`. The native algorithm does not copy the data
but is typically slower.
"""
contract!(α, A::AbstractArray, CA::Symbol, B::AbstractArray, CB::Symbol, β, C::AbstractArray,
    oindA, cindA, oindB, cindB, p1, p2) =
    contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1...,p2...))

# actual implementations for AbstractArray with ind = (indleft..., indright...)

function checked_similar_from_indices(C, ::Type{T}, ind::IndexTuple{N}, A::AbstractArray, CA::Symbol) where {T,N}
    sz = map(n->size(A, n), ind)
    if C !== nothing && isa(C, Array) && sz == size(C) && T == eltype(C)
        return C::Array{T, N}
    else
        return Array{T, N}(undef, sz)
    end
end
function checked_similar_from_indices(C, ::Type{T}, poA::IndexTuple, poB::IndexTuple, ind::IndexTuple{N},
    A::AbstractArray, B::AbstractArray, CA::Symbol, CB::Symbol) where {T,N}

    oszA = map(n->size(A,n), poA)
    oszB = map(n->size(B,n), poB)
    sz = let osz = (oszA..., oszB...)
        map(n->osz[n], ind)
    end
    if C !== nothing && isa(C, Array) && sz == size(C) && T == eltype(C)
        return C::Array{T, N}
    else
        return Array{T, N}(undef, sz)
    end
end

scalar(C::AbstractArray) = ndims(C)==0 ? C[1] : throw(DimensionMismatch())

function add!(α, A::AbstractArray{<:Any, N}, CA::Symbol, β, C::AbstractArray{<:Any, N}, indCinA) where {N}
    N == length(indCinA) || throw(IndexError("Invalid permutation of length $N: $indCinA"))
    if CA == :N
        @unsafe_strided A C _add!(α, A, β, C, (indCinA...,))
    elseif CA == :C
        @unsafe_strided A C _add!(α, conj(A), β, C, (indCinA...,))
    else
        throw(ArgumentError("Unknown conjugation flag: $CA"))
    end
    return C
end
_add!(α, A::UnsafeStridedView{<:Any,N}, β, C::UnsafeStridedView{<:Any,N}, indCinA::IndexTuple{N}) where N =
    LinearAlgebra.axpby!(α, permutedims(A, indCinA), β, C)

function trace!(α, A::AbstractArray{<:Any, NA}, CA::Symbol, β, C::AbstractArray{<:Any, NC},
    indCinA, cindA1, cindA2) where {NA,NC}

    NC == length(indCinA) || throw(IndexError("Invalid selection of $NC out of $NA: $indCinA"))
    NA-NC == 2*length(cindA1) == 2*length(cindA2) || throw(IndexError("invalid number of trace dimension"))
    if CA == :N
        @unsafe_strided A C _trace!(α, A, β, C, (indCinA...,), (cindA1...,), (cindA2...,))
    elseif CA == :C
        @unsafe_strided A C _trace!(α, conj(A), β, C, (indCinA...,), (cindA1...,), (cindA2...,))
    else
        throw(ArgumentError("Unknown conjugation flag: $CA"))
    end
    return C
end
function _trace!(α, A::UnsafeStridedView, β, C::UnsafeStridedView, indCinA::IndexTuple{NC}, cindA1::IndexTuple{NT}, cindA2::IndexTuple{NT}) where {NC,NT}
    sizeA = i->size(A, i)
    strideA = i->stride(A, i)
    tracesize = sizeA.(cindA1)
    tracesize == sizeA.(cindA2) || throw(DimensionMismatch("non-matching trace sizes"))
    size(C) == sizeA.(indCinA) || throw(DimensionMismatch("non-matching sizes"))

    newstrides = (strideA.(indCinA)..., (strideA.(cindA1) .+ strideA.(cindA2))...)
    newsize = (size(C)..., tracesize...)
    A2 = UnsafeStridedView(A.ptr, newsize, newstrides, A.offset, A.op)

    if α != 1
        if β == 0
            Strided._mapreducedim!(x->α*x, +, zero, newsize, (C, A2))
        elseif β == 1
            Strided._mapreducedim!(x->α*x, +, nothing, newsize, (C, A2))
        else
            Strided._mapreducedim!(x->α*x, +, y->β*y, newsize, (C, A2))
        end
    else
        if β == 0
            return Strided._mapreducedim!(identity, +, zero, newsize, (C, A2))
        elseif β == 1
            Strided._mapreducedim!(identity, +, nothing, newsize, (C, A2))
        else
            Strided._mapreducedim!(identity, +, y->β*y, newsize, (C, A2))
        end
    end
    return C
end

function isblascontractable(A::AbstractArray{T,N}, p1::IndexTuple, p2::IndexTuple, C::Symbol) where {T,N}
    T <: LinearAlgebra.BlasFloat || return false
    strideA = let s = strides(Strided.UnsafeStridedView(A))
        i->s[i]
    end
    sizeA = let s = size(A)
        i-> s[i]
    end

    canreshape1, s1 = _canreshape(sizeA.(p1), strideA.(p1))
    canreshape2, s2 = _canreshape(sizeA.(p2), strideA.(p2))

    if C == :D # destination
        return canreshape1 && canreshape2 && s1 == 1
    elseif C == :C # conjugated
        return canreshape1 && canreshape2 && s2 == 1
    else
        return canreshape1 && canreshape2 && (s1 == 1 || s2 == 1)
    end
end
_canreshape(::Tuple{}, ::Tuple{}) = true, 1, ()
function _canreshape(dims::Dims{N}, strides::Dims{N}) where {N}
    t1 = Base.front(dims) .* Base.front(strides)
    t2 = Base.tail(strides)
    return (t1 == t2, strides[1])
end

function unsafe_contract!(α, A::AbstractArray{T}, CA::Symbol, B::AbstractArray{T}, CB::Symbol, β, C::AbstractArray{T},
    oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple, oindAinC::IndexTuple, oindBinC::IndexTuple) where {T<:LinearAlgebra.BlasFloat}

    ndims(A) == length(oindA) + length(cindA) || throw(IndexError("Invalid permutation of $(ndims(A)) indices: $((oindA..., cindA...))"))
    ndims(B) == length(oindB) + length(cindB) || throw(IndexError("Invalid permutation of $(ndims(B)) indices: $((oindB..., cindB...))"))
    ndims(C) == length(oindAinC) + length(oindBinC) || throw(IndexError("Invalid permutation of $(ndims(C)) indices: $((oindAinC..., oindBinC...))"))

    sizeA = i->size(A, i)
    sizeB = i->size(B, i)
    sizeC = i->size(C, i)

    csizeA = sizeA.(cindA)
    csizeB = sizeB.(cindB)
    osizeA = sizeA.(oindA)
    osizeB = sizeB.(oindB)

    csizeA == csizeB || throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    osizeA == sizeC.(oindAinC) || throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
    osizeB == sizeC.(oindBinC) || throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    A2′ = permutedims(UnsafeStridedView(A), (oindA..., cindA...))
    A2 = sreshape(permutedims(UnsafeStridedView(A), (oindA..., cindA...)), (prod(osizeA), prod(csizeA)))
    if CA == :N
        if stride(A2, 1) != 1
            A2 = permutedims(A2, (2,1))
            cA = 'T'
        else
            cA = 'N'
        end
    elseif CA == :C
        A2 = permutedims(A2, (2,1))
        cA = 'C'
    end

    B2 = sreshape(permutedims(UnsafeStridedView(B), (cindB..., oindB...)), (prod(csizeB), prod(osizeB)))
    if CB == :N
        if stride(B2, 1) != 1
            B2 = permutedims(B2, (2,1))
            cB = 'T'
        else
            cB = 'N'
        end
    elseif CB == :C
        B2 = permutedims(B2, (2,1))
        cB = 'C'
    end

    C2 = sreshape(permutedims(UnsafeStridedView(C), (oindAinC..., oindBinC...)), (prod(osizeA), prod(osizeB)))
    LinearAlgebra.BLAS.gemm!(cA,cB, convert(T, α), A2, B2, convert(T, β), C2)

    return C
end

_trivtuple(t::NTuple{N}) where {N} = ntuple(identity, Val(N))

function contract!(α, A::AbstractArray, CA::Symbol, B::AbstractArray, CB::Symbol, β, C::AbstractArray, oindA, cindA, oindB, cindB, indCinoAB)
    pA = (oindA...,cindA...)
    (length(pA) == ndims(A) && TupleTools.isperm(pA)) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
    pB = (oindB...,cindB...)
    (length(pB) == ndims(B) && TupleTools.isperm(pB)) ||
        throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
    (length(oindA) + length(oindB) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (ndims(C) == length(indCinoAB) && isperm(indCinoAB)) ||
        throw(IndexError("invalid permutation of length $(ndims(C)): $indCinoAB"))

    sizeA = i->size(A, i)
    sizeB = i->size(B, i)
    sizeC = i->size(C, i)

    csizeA = sizeA.(cindA)
    csizeB = sizeB.(cindB)
    osizeA = sizeA.(oindA)
    osizeB = sizeB.(oindB)

    csizeA == csizeB || throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    sizeAB = let osize = (osizeA..., osizeB...)
        i->osize[i]
    end
    sizeAB.(indCinoAB) == size(C) || throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    TC = eltype(C)
    if use_blas() && TC <: BlasFloat
        if isblascontractable(A, oindA, cindA, CA) && eltype(A) == TC
            A2 = A
            CA2 = CA
        else
            A2 = similar_from_indices(TC, oindA, cindA, A, CA)
            add!(1, A, CA, 0, A2, oindA, cindA)
            CA2 = :N
            oindA = _trivtuple(oindA)
            cindA = _trivtuple(cindA) .+ length(oindA)
        end
        if isblascontractable(B, cindB, oindB, CB) && eltype(B) == TC
            B2 = B
            CB2 = CB
        else
            B2 = similar_from_indices(TC, cindB, oindB, B, CB)
            add!(1, B, CB, 0, B2, cindB, oindB)
            CB2 = :N
            cindB = _trivtuple(cindB)
            oindB = _trivtuple(oindB) .+ length(cindB)
        end
        ipC = TupleTools.invperm(indCinoAB)
        oindAinC = TupleTools.getindices(ipC, _trivtuple(oindA))
        oindBinC = TupleTools.getindices(ipC, length(oindA) .+ _trivtuple(oindB))
        if isblascontractable(C, oindAinC, oindBinC, :D)
            C2 = C
            unsafe_contract!(α, A2, CA2, B2, CB2, β, C2, oindA, cindA, oindB, cindB, oindAinC, oindBinC)
        else
            C2 = similar_from_indices(TC, oindAinC, oindBinC, C, :N)
            unsafe_contract!(1, A2, CA2, B2, CB2, 0, C2, oindA, cindA, oindB, cindB, _trivtuple(oindA), length(oindA) .+ _trivtuple(oindB))
            add!(α, C2, :N, β, C, indCinoAB, ())
        end
    else
        ipC = TupleTools.invperm(indCinoAB)
        GC.@preserve A B C begin
            AS = sreshape(permutedims(UnsafeStridedView(A), (oindA..., cindA...)), (osizeA..., one.(osizeB)..., csizeA...))
            BS = sreshape(permutedims(UnsafeStridedView(B), (oindB..., cindB...)), (one.(osizeA)..., osizeB..., csizeB...))
            CS = sreshape(permutedims(UnsafeStridedView(C), ipC), (osizeA..., osizeB..., one.(csizeA)...))
            totsize = (osizeA..., osizeB..., csizeA...)
            if α != 1
                if β == 0
                    Strided._mapreducedim!((x,y)->α*x*y, +, zero, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                elseif β == 1
                    Strided._mapreducedim!((x,y)->α*x*y, +, nothing, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                else
                    Strided._mapreducedim!((x,y)->α*x*y, +, y->β*y, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                end
            else
                if β == 0
                    Strided._mapreducedim!(*, +, zero, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                elseif β == 1
                    Strided._mapreducedim!(*, +, nothing, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                else
                    Strided._mapreducedim!(*, +, y->β*y, totsize,
                        (CS, CA == :N ? AS : conj(AS), CB == :N ? BS : conj(BS)))
                end
            end
        end
    end
    return C
end
