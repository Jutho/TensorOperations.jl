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
    tensorcontract!(one(T),A,labelsA,'N',B,labelsB,'N',zero(T),C,outputlabels;method=method)
    return C
end

# In-place method
#-----------------
function tensorcontract!{TA,TB,TC}(alpha::Number,A::StridedArray{TA},labelsA,conjA::Char,B::StridedArray{TB},labelsB,conjB::Char,beta::Number,C::StridedArray{TC},labelsC;method::Symbol=:BLAS)
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
    olabelsA=intersect(ulabelsC,ulabelsA)
    numopenA=length(olabelsA)
    olabelsB=intersect(ulabelsC,ulabelsB)
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
        odimsA[i]==dimC[oindCA[i]] || throw(DimensionMismatch("dimension mismatch for label $(olabelsA[i])"))
    end
    for i=1:numopenB
        odimsB[i]==dimC[oindCB[i]] || throw(DimensionMismatch("dimension mismatch for label $(olabelsB[i])"))
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
        # try to change extra allocation as much as possible
        if vcat(olabelsB,olabelsA)==labelsC[1:NC] # better to change role of A and B
            labelsA,labelsB=labelsB,labelsA
            olabelsA,olabelsB=olabelsB,olabelsA
            odimsA,odimsB=odimsB,odimsA
            cdimsA,cdimsB=cdimsB,cdimsA
            A,B=B,A
            NA,NB=NB,NA
        end

        olengthA=prod(odimsA)
        olengthB=prod(odimsB)
        clength=prod(cdimsA)
        
        # permute A
        if conjA=='C'
            if vcat(clabels,olabelsA)==labelsA[1:NA] && TA==TC && isa(A,Array)
                Amat=A
                Amat=reshape(Amat,(clength,olengthA))
            else
                Amat=similar(A,TC,tuple(cdimsA...,odimsA...))
                tensorcopy!(A,labelsA,Amat,vcat(clabels,olabelsA))
                Amat=reshape(Amat,(clength,olengthA))
            end
        elseif conjA=='N'
            if vcat(olabelsA,clabels)==labelsA[1:NA] && TA==TC && isa(A,Array)
                Amat=A
                Amat=reshape(Amat,(olengthA,clength))
            elseif vcat(clabels,olabelsA)==labelsA[1:NA] && TA==TC && isa(A,Array)
                conjA='T'
                Amat=A
                Amat=reshape(Amat,(clength,olengthA))
            else
                conjA='T' # it is more efficient to compute At*B
                Amat=similar(A,TC,tuple(cdimsA...,odimsA...))
                tensorcopy!(A,labelsA,Amat,vcat(clabels,olabelsA))
                Amat=reshape(Amat,(clength,olengthA))
            end
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end

        # permute B
        if conjB=='C'
            if vcat(olabelsB,clabels)==labelsB[1:NB] && TB==TC && isa(B,Array)
                Bmat=B
                Bmat=reshape(Bmat,(olengthB,clength))
            else
                Bmat=similar(B,TC,tuple(odimsB...,cdimsB...))
                tensorcopy!(B,labelsB,Bmat,vcat(olabelsB,clabels))
                Bmat=reshape(Bmat,(olengthB,clength))
            end
        elseif conjB=='N'
            if vcat(clabels,olabelsB)==labelsB[1:NB] && TB==TC && isa(B,Array)
                Bmat=B
                Bmat=reshape(Bmat,(clength,olengthB))
            elseif vcat(olabelsB,clabels)==labelsB[1:NB] && TB==TC && isa(B,Array)
                conjB='T'
                Bmat=B
                Bmat=reshape(Bmat,(olengthB,clength))
            else
                Bmat=similar(B,TC,tuple(cdimsB...,odimsB...))
                tensorcopy!(B,labelsB,Bmat,vcat(clabels,olabelsB))
                Bmat=reshape(Bmat,(clength,olengthB))
            end
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end
        
        # calculate C
        totalodimsA=prod(odimsA)
        totalodimsB=prod(odimsB)
        totalcdims=prod(cdimsA)
        
        if vcat(olabelsA,olabelsB)==labelsC[1:NC] && isa(C,Array)
            Cmat=reshape(C,(olengthA,olengthB))
            Base.LinAlg.BLAS.gemm!(conjA,conjB,convert(TC,alpha),Amat,Bmat,convert(TC,beta),Cmat)
        else
            Cmat=zeros(TC,(olengthA,olengthB))
            Base.LinAlg.BLAS.gemm!(conjA,conjB,one(TC),Amat,Bmat,zero(TC),Cmat)
            Cmat=reshape(Cmat,tuple(odimsA...,odimsB...))
            tensoradd!(alpha,Cmat,vcat(olabelsA,olabelsB),beta,C,labelsC)
        end
    elseif method==:native
        if NA>=NB # allows to generate less method definitions
            tensorcontract_native!(alpha,A,conjA,B,conjB,beta,C,oindA,cindA,oindB,cindB,oindCA,oindCB)
        else
            tensorcontract_native!(alpha,B,conjB,A,conjA,beta,C,oindB,cindB,oindA,cindA,oindCB,oindCA)
        end
   # elseif method==:buffered
   #      # to be written: allocate a fixed buffer, that can be used as temporary space for BLAS
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

const CONTRACTGENERATE=[(1,1,2), # outer product of 2 vectors
(1,1,0), # scalar product of 2 vectors
(2,1,1), # mat vec multiplication
(1,2,1), # vec mat multiplication
(2,2,4), # mat mat outer product
(2,2,2), # mat mat multiplication
(2,2,0), # mat mat scalar product
(3,1,4),
(3,1,2),
(3,2,5),
(3,2,3),
(3,2,1),
(3,3,6),
(3,3,4),
(3,3,2),
(3,3,0),
(4,1,5),
(4,1,3),
(4,2,6),
(4,2,4),
(4,2,2),
(4,3,5), # restrict to outputs with NC<=6
(4,3,3),
(4,3,1),
(4,4,6),
(4,4,4),
(4,4,2),
(4,4,0)]

macro gencontractkernel(N1,N2,N3,order,alpha,A,conjA,B,conjB,beta,C,startA,startB,startC,odimsA,odimsB,cdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
    _gencontractkernel(N1,N2,N3,order,alpha,A,conjA,B,conjB,beta,C,startA,startB,startC,odimsA,odimsB,cdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
end
function _gencontractkernel(N1::Int,N2::Int,N3::Int,order::Symbol,
    alpha::Symbol,A::Symbol,conjA::Symbol,B::Symbol,conjB::Symbol,beta::Symbol,C::Symbol,
    startA::Symbol,startB::Symbol,startC::Symbol,odimsA::Symbol,odimsB::Symbol,cdims::Symbol,
    ostridesA::Symbol,cstridesA::Symbol,ostridesB::Symbol,cstridesB::Symbol,ostridesCA::Symbol,ostridesCB::Symbol)
    ex=quote
        local indA1, indA2, indB1, indB2, indC1, indC2
        # we still have to implement other orders
#        if $(esc(order))==0 # i,j,k
            @stridedloops($N1, i, $(esc(odimsA)), indA1, $(esc(startA)), $(esc(ostridesA)), indC1, $(esc(startC)), $(esc(ostridesCA)), begin
                @stridedloops($N2, j, $(esc(odimsB)), indB1, $(esc(startB)), $(esc(ostridesB)), indC2, indC1, $(esc(ostridesCB)), begin
                    @inbounds localC=$(esc(beta))*$(esc(C))[indC2]
                    @stridedloops($N3, k, $(esc(cdims)), indA2, indA1, $(esc(cstridesA)), indB2, indB1, $(esc(cstridesB)), begin
                        @inbounds localA=($(esc(conjA))=='C' ? conj($(esc(A))[indA2]) : $(esc(A))[indA2])
                        @inbounds localB=($(esc(conjB))=='C' ? conj($(esc(B))[indB2]) : $(esc(B))[indB2])
                        localC+=$(esc(alpha))*localA*localB
                    end)
                    @inbounds $(esc(C))[indC2]=localC
                end)
            end)
#        end
    end
    ex
end

@mnpgenerate NA NB NC typeof(C) [(2,2,2)] function tensorcontract_native!{TA,TB,TC,NA,NB,NC}(alpha::Number,A::StridedArray{TA,NA},conjA::Char,B::StridedArray{TB,NB},conjB::Char,beta::Number,C::StridedArray{TC,NC},oindA,cindA,oindB,cindB,oindCA,oindCB)
    # only basic checking, this function is not expected to be called directly
    length(oindA)==length(oindCA) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB)==length(oindCB) || throw(DimensionMismatch("invalid contraction pattern"))
    length(cindA)==length(cindB)==div(NA+NB-NC,2) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindA)+length(cindA)==NA || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB)+length(cindB)==NB || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindCA)+length(oindCB)==NC || throw(DimensionMismatch("invalid contraction pattern"))
    
    ostridesA=Int[stride(A,i) for i in oindA]
    cstridesA=Int[stride(A,i) for i in cindA]
    ostridesB=Int[stride(B,i) for i in oindB]
    cstridesB=Int[stride(B,i) for i in cindB]
    ostridesCA=Int[stride(C,i) for i in oindCA]
    ostridesCB=Int[stride(C,i) for i in oindCB]
    
    # calculate optimal contraction order for inner loops
    order=0
    if NA-div(NA+NB-NC,2)>0 && minimum(ostridesA)<minimum(cstridesA)
        order+=1 # k after i
    end
    if NB-div(NA+NB-NC,2)>0 && minimum(ostridesB)<minimum(cstridesB)
        order+=2 # k after j
    end
    if NB-div(NA+NB-NC,2)==0 || (NA-div(NA+NB-NC,2)>0 && minimum(ostridesA)<minimum(ostridesB))
        order+=4 # j after i
    end
    # more to left = later
    # i,j,k: order==0
    # i,k,j: order==2 || order==6
    # j,i,k: order==4
    # j,k,i: order==5 || order==1
    # k,i,j: order==3
    # k,j,i: order==7
    # 1 and 6 are the frustrated cases where no optimal order exists
    
    # calculate dims as variables
    olengthA=1
    olengthB=1
    clength=1
    @nexprs NA-div(NA+NB-NC,2) d->(odimsA_{d}=size(A,oindA[d]);olengthA*=odimsA_{d})
    @nexprs NB-div(NA+NB-NC,2) d->(odimsB_{d}=size(B,oindB[d]);olengthB*=odimsB_{d})
    @nexprs div(NA+NB-NC,2) d->(cdims_{d}=size(A,cindA[d]);clength*=cdims_{d})

    # calculate strides as variables
    @nexprs NA-div(NA+NB-NC,2) d->(begin
        ostridesA_{d}=ostridesA[d]
        ostridesCA_{d}=ostridesCA[d]
        minostridesA_{d}=min(ostridesA_{d},ostridesCA_{d})
    end)
    @nexprs NB-div(NA+NB-NC,2) d->(begin
        ostridesB_{d}=ostridesB[d]
        ostridesCB_{d}=ostridesCB[d]
        minostridesB_{d}=min(ostridesB_{d},ostridesCB_{d})
    end)
    @nexprs div(NA+NB-NC,2) d->(begin
        cstridesA_{d}=cstridesA[d]
        cstridesB_{d}=cstridesB[d]
        mincstrides_{d}=min(cstridesA_{d},cstridesB_{d})
    end)
    
    if beta==zero(beta)
        fill!(C,zero(TC))
    end
    
    startA=1
    local Alinear::Array{TA,NA}
    if isa(A, SubArray)
        startA = A.first_index
        Alinear = A.parent
    else
        Alinear = A
    end
    startB=1
    local Blinear::Array{TB,NB}
    if isa(B, SubArray)
        startB = B.first_index
        Blinear = B.parent
    else
        Blinear = B
    end
    startC=1
    local Clinear::Array{TC,NC}
    if isa(C, SubArray)
        startC = C.first_index
        Clinear = C.parent
    else
        Clinear = C
    end
    
    if olengthA<=2*OBASELENGTH && olengthB<=2*OBASELENGTH && clength<=2*CBASELENGTH
        @gencontractkernel(NA-div(NA+NB-NC,2),NB-div(NA+NB-NC,2),div(NA+NB-NC,2),order,alpha,Alinear,conjA,Blinear,conjB,beta,Clinear,startA,startB,startC,odimsA,odimsB,cdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
    else
        # build recursive stack
        depth=iceil(log2(olengthA/OBASELENGTH))+iceil(log2(olengthB/OBASELENGTH))+iceil(log2(clength/CBASELENGTH))+4 # 4 levels safety margin
        level=1 # level of recursion
        stackstep=zeros(Int,depth) # record step of algorithm at the different recursion level
        stackstep[level]=0
        stackoblengthA=zeros(Int,depth)
        stackoblengthA[level]=olengthA
        stackoblengthB=zeros(Int,depth)
        stackoblengthB[level]=olengthB
        stackcblength=zeros(Int,depth)
        stackcblength[level]=clength
        @nexprs NA-div(NA+NB-NC,2) d->begin
            stackobdimsA_{d} = zeros(Int,depth)
            stackobdimsA_{d}[level] = odimsA_{d}
        end
        @nexprs NB-div(NA+NB-NC,2) d->begin
            stackobdimsB_{d} = zeros(Int,depth)
            stackobdimsB_{d}[level] = odimsB_{d}
        end
        @nexprs div(NA+NB-NC,2) d->begin
            stackcbdims_{d} = zeros(Int,depth)
            stackcbdims_{d}[level] = cdims_{d}
        end
        stackbstartA=zeros(Int,depth)
        stackbstartA[level]=startA
        stackbstartB=zeros(Int,depth)
        stackbstartB[level]=startB
        stackbstartC=zeros(Int,depth)
        stackbstartC[level]=startC
        stackgamma=zeros(typeof(beta),depth)
        stackgamma[level]=beta

        stackdA=zeros(Int,depth)
        stackdB=zeros(Int,depth)
        stackdC=zeros(Int,depth)
        stackdmax=zeros(Int,depth)
        stackwhichd=zeros(Int,depth)
        stacknewdim=zeros(Int,depth)
        stackolddim=zeros(Int,depth)

        while level>0
            step=stackstep[level]
            oblengthA=stackoblengthA[level]
            oblengthB=stackoblengthB[level]
            cblength=stackcblength[level]
            @nexprs NA-div(NA+NB-NC,2) d->(obdimsA_{d} = stackobdimsA_{d}[level])
            @nexprs NB-div(NA+NB-NC,2) d->(obdimsB_{d} = stackobdimsB_{d}[level])
            @nexprs div(NA+NB-NC,2) d->(cbdims_{d} = stackcbdims_{d}[level])
            bstartA=stackbstartA[level]
            bstartB=stackbstartB[level]
            bstartC=stackbstartC[level]
            gamma=stackgamma[level]

            if (oblengthA<=OBASELENGTH && oblengthB<=OBASELENGTH && cblength<=CBASELENGTH) || level==depth # base case
                @gencontractkernel(NA-div(NA+NB-NC,2),NB-div(NA+NB-NC,2),div(NA+NB-NC,2),order,alpha,Alinear,conjA,Blinear,conjB,gamma,Clinear,bstartA,bstartB,bstartC,obdimsA,obdimsB,cbdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
                level-=1
            elseif step==0
                # find which dimension to divide
                dmax=0
                whichd=0
                maxval=0
                newdim=0
                olddim=0
                dA=0
                dB=0
                dC=0
                if oblengthA>=oblengthB && oblengthA*CBASELENGTH>=cblength*OBASELENGTH
                    whichd=1
                    @nexprs NA-div(NA+NB-NC,2) d->begin
                        newmax=obdimsA_{d}*minostridesA_{d}
                        if obdimsA_{d}>1 && newmax>maxval
                            dmax=d
                            olddim=obdimsA_{d}
                            newdim=olddim>>1
                            dA=ostridesA_{d}
                            dB=0
                            dC=ostridesCA_{d}
                            maxval=newmax
                        end
                    end
                elseif oblengthB>=oblengthA && oblengthB*CBASELENGTH>=cblength*OBASELENGTH
                    whichd=2
                    @nexprs NB-div(NA+NB-NC,2) d->begin
                        newmax=obdimsB_{d}*minostridesB_{d}
                        if obdimsB_{d}>1 && newmax>maxval
                            dmax=d
                            olddim=obdimsB_{d}
                            newdim=olddim>>1
                            dA=0
                            dB=ostridesB_{d}
                            dC=ostridesCB_{d}
                            maxval=newmax
                        end
                    end
                else
                    whichd=3
                    @nexprs div(NA+NB-NC,2) d->begin
                        newmax=cbdims_{d}*mincstrides_{d}
                        if cbdims_{d}>1 && newmax>maxval
                            dmax=d
                            olddim=cbdims_{d}
                            newdim=olddim>>1
                            dA=cstridesA_{d}
                            dB=cstridesB_{d}
                            dC=0
                            maxval=newmax
                        end
                    end
                end
                stackolddim[level]=olddim
                stacknewdim[level]=newdim
                stackdmax[level]=dmax
                stackwhichd[level]=whichd
                stackdA[level]=dA
                stackdB[level]=dB
                stackdC[level]=dC

                stackstep[level+1]=0
                stackoblengthA[level+1]= (whichd==1 ? div(oblengthA,olddim)*newdim : oblengthA)
                stackoblengthB[level+1]= (whichd==2 ? div(oblengthB,olddim)*newdim : oblengthB)
                stackcblength[level+1]= (whichd==3 ? div(cblength,olddim)*newdim : cblength)
                @nexprs NA-div(NA+NB-NC,2) d->(stackobdimsA_{d}[level+1] = (d==dmax && whichd==1 ? newdim : obdimsA_{d}))
                @nexprs NB-div(NA+NB-NC,2) d->(stackobdimsB_{d}[level+1] = (d==dmax && whichd==2 ? newdim : obdimsB_{d}))
                @nexprs div(NA+NB-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax && whichd==3 ? newdim : cbdims_{d}))
                stackbstartA[level+1]=bstartA
                stackbstartB[level+1]=bstartB
                stackbstartC[level+1]=bstartC
                stackgamma[level+1]=gamma

                stackstep[level]+=1
                level+=1
            elseif step==1
                dmax=stackdmax[level]
                whichd=stackwhichd[level]
                olddim=stackolddim[level]
                newdim=stacknewdim[level]
                dA=stackdA[level]
                dB=stackdB[level]
                dC=stackdC[level]

                stackstep[level+1]=0
                stackoblengthA[level+1]= (whichd==1 ? div(oblengthA,olddim)*(olddim-newdim) : oblengthA)
                stackoblengthB[level+1]= (whichd==2 ? div(oblengthB,olddim)*(olddim-newdim) : oblengthB)
                stackcblength[level+1]= (whichd==3 ? div(cblength,olddim)*(olddim-newdim) : cblength)
                @nexprs NA-div(NA+NB-NC,2) d->(stackobdimsA_{d}[level+1] = (d==dmax && whichd==1 ? olddim-newdim : obdimsA_{d}))
                @nexprs NB-div(NA+NB-NC,2) d->(stackobdimsB_{d}[level+1] = (d==dmax && whichd==2 ? olddim-newdim : obdimsB_{d}))
                @nexprs div(NA+NB-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax && whichd==3 ? olddim-newdim : cbdims_{d}))
                stackbstartA[level+1]=bstartA+dA*newdim
                stackbstartB[level+1]=bstartB+dB*newdim
                stackbstartC[level+1]=bstartC+dC*newdim
                stackgamma[level+1]=(whichd==1 || whichd==2 ? gamma : one(gamma))

                stackstep[level]+=1
                level+=1
            else
                level-=1
            end
        end
    end
    return C
end