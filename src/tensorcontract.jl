# tensorcontract.jl
#
# Method for contracting two tensors and adding the result
# to a third tensor, according to the specified labels.

# Simple method
#---------------
function tensorcontract{T1,T2}(A::StridedArray{T1},labelsA,B::StridedArray{T2},labelsB,outputlabels=symdiff(labelsA,labelsB);method::Symbol=:BLAS)
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
    return tensorcontract!(one(T),A,labelsA,'N',B,labelsB,'N',zero(T),C,outputlabels;method=method)
end

# In-place method
#-----------------
function tensorcontract!{R,S,T}(alpha::Number,A::StridedArray{S},labelsA,conjA::Char,B::StridedArray{T},labelsB,conjB::Char,beta::Number,C::StridedArray{R},labelsC;method::Symbol=:BLAS)
    # Updates C as beta*C+alpha*contract(A,B), whereby the contraction pattern
    # is specified by labelsA, labelsB and labelsC. The iterables labelsA(B,C)
    # should contain a unique label for every index of array A(B,C), such that
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
    # -> method=:BLAS : permutes tensors (requires extra memory) and then
    #                   calls built-in (typically BLAS) multiplication
    # -> method=:native : memory-free native julia tensor contraction
    # -> method=:buffered : uses memory buffer (not implemented yet)
    
    # Get properties of input arrays
    NA=ndims(A)
    NB=ndims(B)
    NC=ndims(C)

    # Process labels, do some error checking and analyse problem structure
    #----------------------------------------------------------------------
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(LabelError("invalid label specification"))
    end
    ulabelsA=unique(labelsA)
    ulabelsB=unique(labelsB)
    ulabelsC=unique(labelsC)
    if NA!=length(ulabelsA) || NB!=length(ulabelsB) || NC!=length(ulabelsC)
        throw(LabelError("tensorcontract requires unique label for every index of the tensor, handle inner contraction first with tensortrace"))
    end

    clabels=intersect(ulabelsA,ulabelsB)
    numcontract=length(clabels)
    olabelsA=intersect(ulabelsA,ulabelsC)
    numopenA=length(olabelsA)
    olabelsB=intersect(ulabelsB,ulabelsC)
    numopenB=length(olabelsB)
    
    if numcontract+numopenA!=NA || numcontract+numopenB!=NB || numopenA+numopenB!=NC
        throw(LabelError("invalid contraction pattern"))
    end
    
    # Compute and contraction indices and check size compatibility
    #--------------------------------------------------------------
    cindA=indexin(clabels,ulabelsA)
    oindA=indexin(olabelsA,ulabelsA)
    oindCA=indexin(olabelsA,ulabelsC)
    cindB=indexin(clabels,ulabelsB)
    oindB=indexin(olabelsB,ulabelsB)
    oindCB=indexin(olabelsB,ulabelsC)

    # check size compatibility
    dimA=size(A)
    dimB=size(B)
    dimC=size(C)

    cdimsA=dimA[cindA]
    cdimsB=dimB[cindB]
    odimsA=dimA[oindA]
    odimsB=dimB[oindB]
    
    for i=1:numcontract
        cdimsA[i]==cdimsB[i] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    for i=1:numopenA
        odimsA[i]==dimC[oindCA[i]] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    for i=1:numopenB
        odimsB[i]==dimC[oindCB[i]] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    
    # Perform contraction
    
    # The :BLAS method specification permutes A and B such that indopen and
    # indcontract are grouped, reshape them to matrices with all indopen on one
    # side and all indcontract on the other. Compute the data for C from
    # multiplying these matrices. Permute again to bring indices in requested
    # order.
    # While this can potentially use the highly optimized BLAS matrix
    # multiplication, it needs three temporary arrays containing the
    # permuted copies of A, B and C. Memorywise this is far from optimal, and
    # the memory allocation and copying also impacts the computation time.
    
    if method==:BLAS
        # permute A
        if conjA=='C'
            Anew=similar(A,eltype(C),tuple(cdimsA...,odimsA...))
            tensorcopy!(A,labelsA,Anew,vcat(clabels,olabelsA))
        elseif conjA=='N' && conjB=='C' # temporary fix untill At_mul_Bc! is implemented
            conjA='N'
            Anew=similar(A,eltype(C),tuple(odimsA...,cdimsA...))
            tensorcopy!(A,labelsA,Anew,vcat(olabelsA,clabels))
        elseif conjA=='N'
            conjA='T' # it is more efficient to compute At*B
            Anew=similar(A,eltype(C),tuple(cdimsA...,odimsA...))
            tensorcopy!(A,labelsA,Anew,vcat(clabels,olabelsA))
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end

        # permute B
        if conjB=='C'
            Bnew=similar(B,eltype(C),tuple(odimsB...,cdimsB...))
            tensorcopy!(B,labelsB,Bnew,vcat(olabelsB,clabels))
        elseif conjB=='N'
            Bnew=similar(B,eltype(C),tuple(cdimsB...,odimsB...))
            tensorcopy!(B,labelsB,Bnew,vcat(clabels,olabelsB))
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end
        
        # calculate C
        totalodimsA=prod(odimsA)
        totalodimsB=prod(odimsB)
        totalcdims=prod(cdimsA)
        
        Cnew=similar(C,tuple(odimsA...,odimsB...))
        if conjA=='T' && conjB=='N'
            At_mul_B!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalcdims,totalodimsB)))
        elseif conjA=='C' && conjB=='N'
            Ac_mul_B!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalcdims,totalodimsB)))
        elseif conjA=='N' && conjB=='C'
            A_mul_Bc!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalodimsA,totalcdims)),reshape(Bnew,(totalodimsB,totalcdims)))
        else
            Ac_mul_Bc!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalodimsB,totalcdims)))
        end
        tensoradd!(alpha,Cnew,vcat(olabelsA,olabelsB),beta,C,labelsC)
        return C
    elseif method==:native
        stridesA=strides(A)
        stridesB=strides(B)
        stridesC=strides(C)
        ostridesA=stridesA[oindA]
        cstridesA=stridesA[cindA]
        ostridesB=stridesB[oindB]
        cstridesB=stridesB[cindB]
        ostridesCA=stridesC[oindCA]
        ostridesCB=stridesC[oindCB]

        unsafe_tensorcontract!(odimsA,odimsB,cdimsA,convert(R,alpha),pointer(A),conjA,ostridesA,cstridesA,pointer(B),conjB,ostridesB,cstridesB,convert(R,beta),pointer(C),ostridesCA,ostridesCB)
        return C
        
    else method==:buffered
        # to be written: allocate a fixed buffer, that can be used as temporary space for BLAS
    end
end

# Low-level method
#------------------
let _tensorcontract_defined=Dict{(Int,Int,Int), Bool}()
    global unsafe_tensorcontract!
    function unsafe_tensorcontract!{T,TA,TB,N1,N2,N3}(odimsA::NTuple{N1,Int},odimsB::NTuple{N2,Int},cdims::NTuple{N3,Int},alpha::T,A::Ptr{TA},conjA::Char,ostridesA::NTuple{N1,Int},cstridesA::NTuple{N3,Int},B::Ptr{TB},conjB::Char,ostridesB::NTuple{N2,Int},cstridesB::NTuple{N3,Int},beta::T,C::Ptr{T},ostridesCA::NTuple{N1,Int},ostridesCB::NTuple{N2,Int},obdimsA::NTuple{N1,Int},obdimsB::NTuple{N2,Int},cbdims::NTuple{N3,Int})
        def=get(_tensorcontract_defined,(N1,N2,N3),false)
        if !def
            ex=quote
            function _unsafe_tensorcontract!{T,TA,TB}(odimsA::NTuple{$N1,Int},odimsB::NTuple{$N2,Int},cdims::NTuple{$N3,Int},alpha::T,A::Ptr{TA},conjA::Char,ostridesA::NTuple{$N1,Int},cstridesA::NTuple{$N3,Int},B::Ptr{TB},conjB::Char,ostridesB::NTuple{$N2,Int},cstridesB::NTuple{$N3,Int},beta::T,C::Ptr{T},ostridesCA::NTuple{$N1,Int},ostridesCB::NTuple{$N2,Int},obdimsA::NTuple{$N1,Int},obdimsB::NTuple{$N2,Int},cbdims::NTuple{$N3,Int})
                # check conjugation input
                conjA=='N' || conjA=='C' || throw(ArgumentError("invalid conjugation specification"))
                conjB=='N' || conjB=='C' || throw(ArgumentError("invalid conjugation specification"))
                
                # calculate dims as variables
                @nexprs $N1 d->(odimsA_{d}=odimsA[d])
                @nexprs $N2 d->(odimsB_{d}=odimsB[d])
                @nexprs $N3 d->(cdims_{d}=cdims[d])
                @nexprs $N1 d->(obdimsA_{d}=obdimsA[d])
                @nexprs $N2 d->(obdimsB_{d}=obdimsB[d])
                @nexprs $N3 d->(cbdims_{d}=cbdims[d])
                # calculate strides as variables
                @nexprs $N1 d->(ostridesA_{d}=ostridesA[d])
                @nexprs $N3 d->(cstridesA_{d}=cstridesA[d])
                @nexprs $N2 d->(ostridesB_{d}=ostridesB[d])
                @nexprs $N3 d->(cstridesB_{d}=cstridesB[d])
                @nexprs $N1 d->(ostridesCA_{d}=ostridesCA[d])
                @nexprs $N2 d->(ostridesCB_{d}=ostridesCB[d])
    
                @nexprs 1 d->(indA1_{$N1}=1)
                @nexprs 1 d->(indC1_{$N1}=1)
                @nloops($N1, outeri, d->1:obdimsA_{d}:odimsA_{d},
                    d->(indA1_{d-1}=indA1_{d};indC1_{d-1}=indC1_{d};ilim_{d}=min(outeri_{d}+obdimsA_{d}-1,odimsA_{d})), # PRE
                    d->(indA1_{d}+=obdimsA_{d}*ostridesA_{d};indC1_{d}+=obdimsA_{d}*ostridesCA_{d}), # POST
                    begin # BODY
                        @nexprs 1 e->(indC2_{$N2}=indC1_0)
                        @nexprs 1 e->(indB1_{$N2}=1)
                        @nloops($N2, outerj, e->1:obdimsB_{e}:odimsB_{e},
                            e->(indC2_{e-1}=indC2_{e};indB1_{e-1}=indB1_{e};jlim_{e}=min(outerj_{e}+obdimsB_{e}-1,odimsB_{e})), # PRE
                            e->(indC2_{e}+=obdimsB_{e}*ostridesCB_{e};indB1_{e}+=obdimsB_{e}*ostridesB_{e}), # POST
                            begin # BODY
                                @nexprs 1 f->(indA2_{$N3}=indA1_0)
                                @nexprs 1 f->(indB2_{$N3}=indB1_0)
                                gamma=beta # for the first value of outerk gamma=beta, afterwards gamma=1
                                @nloops($N3, outerk, f->1:cbdims_{f}:cdims_{f},
                                    f->(indA2_{f-1}=indA2_{f};indB2_{f-1}=indB2_{f};klim_{f}=min(outerk_{f}+cbdims_{f}-1,cdims_{f})), # PRE
                                    f->(indA2_{f}+=cbdims_{f}*cstridesA_{f};indB2_{f}+=cbdims_{f}*cstridesB_{f}), # POST
                                    begin # BODY
                                        @nexprs 1 b->(indB3_{$N2}=indB2_0)
                                        @nexprs 1 b->(indC3_{$N2}=indC2_0)
                                        @nloops($N2, innerj, b->outerj_{b}:jlim_{b},
                                            b->(indB3_{b-1}=indB3_{b};indC3_{b-1}=indC3_{b}), # PRE
                                            b->(indB3_{b}+=ostridesB_{b};indC3_{b}+=ostridesCB_{b}), # POST
                                            begin # BODY
                                                @nexprs 1 c->(indA3_{$N3}=indA2_0)
                                                @nexprs 1 c->(indB4_{$N3}=indB3_0)
                                                delta=gamma # for the first iteration of innerk delta=gamma, afterwards delta=1
                                                @nloops($N3, innerk, c->outerk_{c}:klim_{c},
                                                    c->(indA3_{c-1}=indA3_{c};indB4_{c-1}=indB4_{c}), # PRE
                                                    c->(indA3_{c}+=cstridesA_{c};indB4_{c}+=cstridesB_{c}), # POST
                                                    begin # BODY
                                                        localB=(conjB=='C' ? conj(unsafe_load(B,indB4_0)) : unsafe_load(B,indB4_0))
                                                        @nexprs 1 a->(indA4_{$N1}=indA3_0)
                                                        @nexprs 1 a->(indC4_{$N1}=indC3_0)
                                                        @nloops($N1, inneri, a->outeri_{a}:ilim_{a},
                                                            a->(indA4_{a-1}=indA4_{a};indC4_{a-1}=indC4_{a}), # PRE
                                                            a->(indA4_{a}+=ostridesA_{a};indC4_{a}+=ostridesCA_{a}), # POST
                                                            begin # BODY
                                                                localA=(conjA=='C' ? conj(unsafe_load(A,indA4_0)) : unsafe_load(A,indA4_0))
                                                                localC::T=delta*unsafe_load(C,indC4_0)
                                                                localC+=alpha*localA*localB
                                                                unsafe_store!(C,localC,indC4_0)
                                                            end)
                                                        delta=one(T)
                                                    end)
                                            end)
                                        gamma=one(T)
                                    end)
                            end)
                    end)
                return C
            end
            end
            eval(ex)
            _tensorcontract_defined[(N1,N2,N3)]=true
        end
        _unsafe_tensorcontract!(odimsA,odimsB,cdims,alpha,A,conjA,ostridesA,cstridesA,B,conjB,ostridesB,cstridesB,beta,C,ostridesCA,ostridesCB,obdimsA,obdimsB,cbdims)
    end
end
function unsafe_tensorcontract!{T,TA,TB,N1,N2,N3}(odimsA::NTuple{N1,Int},odimsB::NTuple{N2,Int},cdims::NTuple{N3,Int},alpha::T,A::Ptr{TA},conjA::Char,ostridesA::NTuple{N1,Int},cstridesA::NTuple{N3,Int},B::Ptr{TB},conjB::Char,ostridesB::NTuple{N2,Int},cstridesB::NTuple{N3,Int},beta::T,C::Ptr{T},ostridesCA::NTuple{N1,Int},ostridesCB::NTuple{N2,Int})
    obdimsA,obdimsB,cbdims=blockdims3(odimsA,odimsB,cdims,sizeof(T),ostridesCA,ostridesCB,sizeof(TA),ostridesA,cstridesA,sizeof(TB),ostridesB,cstridesB)
    return unsafe_tensorcontract!(odimsA,odimsB,cdims,alpha,A,conjA,ostridesA,cstridesA,B,conjB,ostridesB,cstridesB,beta,C,ostridesCA,ostridesCB,obdimsA,obdimsB,cbdims)
end