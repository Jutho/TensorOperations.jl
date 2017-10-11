# implementation/indices.jl
#
# Implements the index calculations, i.e. converting the tensor labels into
# indices specifying the operations
import Base.tail

# tuple setdiff, assumes b is completely contained in a or throws error
tsetdiff(a::Tuple, b::Tuple{}) = a
tsetdiff(a::Tuple{Any}, b::Tuple{Any}) = ()
tsetdiff(a::Tuple, b::Tuple{Any}) = a[1] == b[1] ? tail(a) : (a[1], tsetdiff(tail(a), b)...)
tsetdiff(a::Tuple, b::Tuple) = tsetdiff(tsetdiff(a, (b[1],)), tail(b))

# tuple unique: assumes that every element appears exactly twice
tunique(src::Tuple) = tunique(src, ())
tunique(src::NTuple{N,Any}, dst::NTuple{N,Any}) where {N} = dst
tunique(src::Tuple, dst::Tuple) = src[1] in dst ? tunique((tail(src)..., src[1]), dst) : tunique(tail(src), (dst..., src[1]))

# Extract index information
#---------------------------
function add_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}
    indCinA = map(l->findfirst(equalto(l), IA), IC)
    (NA == NC && isperm(indCinA)) || throw(IndexError("invalid index specification: $IA to $IC"))
    return indCinA
end

function trace_indices(IA::NTuple{NA,Any}, IC::NTuple{NC,Any}) where {NA,NC}
    # trace indices
    isodd(length(IA)-length(IC)) && throw(IndexError("invalid trace specification: $IA to $IC"))
    Itrace = tunique(tsetdiff(IA, IC))

    cindA1 = map(l->findfirst(equalto(l), IA), Itrace)
    cindA2 = map(l->findnext(equalto(l), IA, findfirst(equalto(l), IA)+1), Itrace)
    indCinA = map(l->findfirst(equalto(l), IA), IC)

    pA = (indCinA..., cindA1..., cindA2...)
    (isperm(pA) && length(pA) == NA) || throw(IndexError("invalid trace specification: $IA to $IC"))
    return indCinA, cindA1, cindA2
end

function contract_indices(IA::NTuple{NA,Any}, IB::NTuple{NB,Any}, IC::NTuple{NC,Any}) where {NA,NB,NC}
    # labels
    IAB = (IA..., IB...)
    isodd(length(IAB)-length(IC)) && throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))
    Icontract = tunique(tsetdiff(IAB, IC))
    IopenA = tsetdiff(IA, Icontract)
    IopenB = tsetdiff(IB, Icontract)

    # to indices
    cindA = map(l->findfirst(equalto(l), IA), Icontract)
    cindB = map(l->findfirst(equalto(l), IB), Icontract)
    oindA = map(l->findfirst(equalto(l), IA), IopenA)
    oindB = map(l->findfirst(equalto(l), IB), IopenB)
    indCinoAB = map(l->findfirst(equalto(l), (IopenA..., IopenB...)), IC)

    if !isperm((oindA..., cindA...)) || !isperm((oindB..., cindB...)) || !isperm(indCinoAB)
        throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))
    end

    return oindA, cindA, oindB, cindB, indCinoAB
end
