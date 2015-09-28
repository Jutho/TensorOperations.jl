# implementation/indices.jl
#
# Implements the index calculations, i.e. converting the tensor labels into
# indices specifying the operations

# Extract index information
#---------------------------
function add_indices(IA, IC)
    indCinA::Vector{Int}=indexin(collect(IC), collect(IA))
    isperm(indCinA) || throw(IndexError("invalid index specification: $IA to $IC"))
    return indCinA
end

function trace_indices(IA, IC)
    indCinA::Vector{Int}=indexin(collect(IC), collect(IA))

    cI=unique(setdiff(IA, IC))
    cindA1=Array(Int, length(cI))
    cindA2=Array(Int, length(cI))
    for i=1:length(cI)
        cindA1[i] = findfirst(IA, cI[i])
        cindA2[i] = findnext(IA, cI[i], cindA1[i]+1)
        findnext(IA, cI[i], cindA2[i]+1)==0 || throw(IndexError("invalid trace specification: $IA to $IC"))
    end
    pA = vcat(indCinA, cindA1, cindA2)
    isperm(pA) || throw(IndexError("invalid trace specification: $IA to $IC"))
    return indCinA, cindA1, cindA2
end

function contract_indices(IA, IB, IC)
    # Compute contraction indices and check for valid permutation
    NA = length(IA)
    NB = length(IB)
    NC = length(IC)

    NA == length(unique(IA)) || throw(IndexError("handle partial trace first: $IA"))
    NB == length(unique(IB)) || throw(IndexError("handle partial trace first: $IB"))
    NC == length(unique(IC)) || throw(IndexError("handle partial trace first: $IC"))

    cI = intersect(IA, IB)
    cN = length(cI)
    oIA = intersect(IC, IA)
    oNA = length(oIA)
    oIB = intersect(IC, IB)
    oNB = length(oIB)

    if cN+oNA != NA || cN+oNB != NB || oNA+oNB != NC
        throw(IndexError("invalid contraction pattern: $IA * $IB to $IC"))
    end

    cindA::Vector{Int} = indexin(cI, collect(IA))
    oindA::Vector{Int} = indexin(oIA, collect(IA))
    cindB::Vector{Int} = indexin(cI, collect(IB))
    oindB::Vector{Int} = indexin(oIB, collect(IB))
    indCinoAB::Vector{Int} = indexin(collect(IC), vcat(oIA, oIB))

    if !isperm(vcat(oindA, cindA)) || !isperm(vcat(oindB, cindB)) || !isperm(indCinoAB)
        throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))
    end

    return oindA, cindA, oindB, cindB, indCinoAB
end
