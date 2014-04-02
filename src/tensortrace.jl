# tensorcopy.jl
#
# Method for tracing some of the indices of a tensor and
# adding the result to another tensor.


# Simple method
#---------------
function tensortrace{T}(A::StridedArray{T},labelsA,outputlabels)
    dimsA=size(A)
    indCinA=indexin(outputlabels,labelsA)
    if any(indCinA.==0)
        throw(LabelError("invalid label specification"))
    end
    C=similar(A,dimsA[indCinA])
    fill!(C,zero(T))
    return tensortrace!(one(T),A,labelsA,zero(T),C,outputlabels)
end
function tensortrace(A::StridedArray,labelsA) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,2*length(ulabelsA)-length(labelsA))
    i=1
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            labelsC[i]=ulabelsA[j]
            i+=1
        end
    end
    tensortrace(A,labelsA,labelsC)
end

# In place method
#-----------------
function tensortrace!{T1,T2}(alpha::Number,A::StridedArray{T1},labelsA,beta::Number,C::StridedArray{T2},labelsC)
    if length(labelsA)!=ndims(A) || length(labelsC)!=ndims(C)
        throw(LabelError("invalid label specification"))
    end
    
    po=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    pc1=indexin(clabels,labelsA)
    pc2=similar(pc1)
    for i=1:length(clabels)
        pc1[i]=findfirst(labelsA,clabels[i])
        pc2[i]=findnext(labelsA,clabels[i],pc1[i]+1)
    end
    isperm(vcat(po,pc1,pc2)) || throw(LabelError("invalid label specification"))
    
    dims=size(A)
    for i = 1:length(po)
        dims[po[i]] == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    for i = 1:length(pc1)
        dims[pc1[i]] == dims[pc2[i]] || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    
    stridesA=strides(A)
    ostridesA=stridesA[po]
    cstridesA1=stridesA[pc1]
    cstridesA2=stridesA[pc2]
    ostridesC=strides(C)
    
    odims=dims[po]
    cdims=dims[pc1]
    
    unsafe_tensortrace!(odims,cdims,convert(T2,alpha),pointer(A),ostridesA,cstridesA1,cstridesA2,convert(T2,beta),pointer(C),ostridesC)
    return C
end

# Low-level method
#------------------
let _tensortrace_defined=Dict{(Int,Int), Bool}()
    global unsafe_tensortrace!
    function unsafe_tensortrace!{T,TA,N1,N2}(odims::NTuple{N1,Int},cdims::NTuple{N2,Int},alpha::T,A::Ptr{TA},ostridesA::NTuple{N1,Int},cstridesA1::NTuple{N2,Int},cstridesA2::NTuple{N2,Int},beta::T,C::Ptr{T},ostridesC::NTuple{N1,Int},obdims::NTuple{N1,Int}=blockdims1(odims,sizeof(TA),ostridesA,sizeof(T),ostridesC))
        def=get(_tensortrace_defined,(N1,N2),false)
        if !def
            ex=quote
            function _unsafe_tensortrace!{T,TA}(odims::NTuple{$N1,Int},cdims::NTuple{$N2,Int},alpha::T,A::Ptr{TA},ostridesA::NTuple{$N1,Int},cstridesA1::NTuple{$N2,Int},cstridesA2::NTuple{$N2,Int},beta::T,C::Ptr{T},ostridesC::NTuple{$N1,Int},obdims::NTuple{$N1,Int})
                # calculate dims as variables
                @nexprs $N1 d->(odims_{d}=odims[d])
                @nexprs $N2 d->(cdims_{d}=cdims[d])
                @nexprs $N1 d->(obdims_{d}=obdims[d])
                # calculate strides as variables
                @nexprs $N1 d->(ostridesA_{d}=ostridesA[d])
                @nexprs $N1 d->(ostridesC_{d}=ostridesC[d])
                @nexprs $N2 d->(cstridesA1_{d}=cstridesA1[d])
                @nexprs $N2 d->(cstridesA2_{d}=cstridesA2[d])
    
                @nexprs 1 d->(indA1_{$N2}=1)
                @nloops($N2,j,d->1:cdims_{d},
                    d->(indA1_{d-1}=indA1_{d}), # PRE
                    d->(indA1_{d}+=cstridesA1_{d}+cstridesA2_{d}), # POST
                    begin
                        @nexprs 1 e->(indA2_{$N1}=indA1_0)
                        @nexprs 1 e->(indC1_{$N1}=1)
                        @nloops($N1, outeri, e->1:obdims_{e}:odims_{e},
                            e->(indA2_{e-1}=indA2_{e};indC1_{e-1}=indC1_{e};ilim_{e}=min(outeri_{e}+obdims_{e}-1,odims_{e})), # PRE
                            e->(indA2_{e}+=obdims_{e}*ostridesA_{e};indC1_{e}+=obdims_{e}*ostridesC_{e}), # POST
                            begin
                                @nexprs 1 f->(indA3_{$N1}=indA2_0)
                                @nexprs 1 f->(indC2_{$N1}=indC1_0)
                                @nloops($N1, inneri, f->outeri_{f}:ilim_{f},
                                    f->(indA3_{f-1}=indA3_{f};indC2_{f-1}=indC2_{f}), # PRE
                                    f->(indA3_{f}+=ostridesA_{f};indC2_{f}+=ostridesC_{f}), # POST
                                    begin
                                        localC::T=beta*unsafe_load(C,indC2_0)
                                        localC+=alpha*unsafe_load(A,indA3_0) # BODY
                                        unsafe_store!(C,localC,indC2_0)
                                    end)
                            end)
                        beta=one(T)
                    end)
                return C
            end
            end
            eval(ex)
            _tensortrace_defined[(N1,N2)]=true
        end
        _unsafe_tensortrace!(odims,cdims,alpha,A,ostridesA,cstridesA1,cstridesA2,beta,C,ostridesC,obdims)
    end
end