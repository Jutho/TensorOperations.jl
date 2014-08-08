# tensorproduct.jl
#
# Method for creating the tensorproduct of two tensors.

# Simple method
#---------------
function tensorproduct(A::StridedArray,labelsA,B::StridedArray,labelsB,outputlabels=vcat(labelsA,labelsB))
    dimsA=size(A)
    dimsB=size(B)
    dimsC=tuple(dimsA...,dimsB...)
    dimsC=dimsC[indexin(outputlabels,vcat(labelsA,labelsB))]
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,dimsC)
    tensorproduct!(one(T),A,labelsA,B,labelsB,zero(T),C,outputlabels)
end

# In-place method
#-----------------
function tensorproduct!(alpha::Number,A::StridedArray,labelsA,B::StridedArray,labelsB,beta::Number,C::StridedArray,labelsC)
    # Updates C as beta*C+alpha*tensorproduct(A,B), whereby the order of indices
    # in A, B and C are specified by the labels.

    # Get properties of input arrays
    NA=ndims(A)
    NB=ndims(B)
    NC=ndims(C)

    # Process labels, do some error checking and analyse problem structure
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(LabelError("invalid label specification"))
    end
    NC==NA+NB || throw(LabelError("not a valid tensor product specification"))
    labels=setdiff(labelsC,labelsA)
    labels=setdiff(labels,labelsB)

    isempty(labels) || throw(LabelError("not a valid tensor product specification"))

    return tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method=:native)
end
