# tensorproduct.jl
#
# Method for creating the tensorproduct of two tensors.

# Simple method
#---------------
function tensorproduct{T1,T2}(A::StridedArray{T1},labelsA,B::StridedArray{T2},labelsB,outputlabels=vcat(labelsA,labelsB))
    dimsA=size(A)
    dimsB=size(B)

    dimsC=Array(Int,length(outputlabels))
    for (i,l)=enumerate(outputlabels)
        ind=findfirst(labelsA,l)
        if ind>0
            dimsC[i]=dimsA[ind]
        else
            ind=findfirst(labelsB,l)
            if ind>0
                dimsC[i]=dimsB[ind]
            else
                throw(LabelError("invalid label specification"))
            end
        end
    end
    T=promote_type(T1,T2)
    C=similar(A,T,tuple(dimsC...))
    fill!(C,zero(T))
    tensorproduct!(one(T),A,labelsA,B,labelsB,zero(T),C,outputlabels)
    return C
end

# In-place method
#-----------------
function tensorproduct!{TA,TB,TC}(alpha::Number,A::StridedArray{TA},labelsA,B::StridedArray{TB},labelsB,beta::Number,C::StridedArray{TC},labelsC)
    # Updates C as beta*C+alpha*tensorproduct(A,B), whereby the order of indices
    # in A, B and C are specified by the labels.

    # Get properties of input arrays
    NA=ndims(A)
    NB=ndims(B)
    NC=ndims(C)

    # Process labels, do some error checking and analyse problem structure
    #----------------------------------------------------------------------
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(LabelError("invalid label specification"))
    end
    NC==NA+NB || throw(LabelError("not a valid tensor product specification"))
    labels=setdiff(labelsC,labelsA)
    labels=setdiff(labels,labelsB)

    isempty(labels) || throw(LabelError("not a valid tensor product specification"))

    return tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method=:native)
end
