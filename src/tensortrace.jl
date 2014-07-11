# tensorcopy.jl
#
# Method for tracing some of the indices of a tensor and
# adding the result to another tensor.


# Simple method
#---------------
function tensortrace(A::StridedArray,labelsA,outputlabels;basesize::Int=1024)
    T=eltype(A)
    dimsA=size(A)
    indCinA=indexin(outputlabels,labelsA)
    if any(indCinA.==0)
        throw(LabelError("invalid label specification"))
    end
    C=similar(A,dimsA[indCinA])
    fill!(C,zero(T))
    return tensortrace!(one(T),A,labelsA,zero(T),C,outputlabels,basesize)
end
function tensortrace(A::StridedArray,labelsA;basesize::Int=1024) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,0)
    sizehint(labelsC,length(labelsA))
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            push!(labelsC,ulabelsA[j])
        end
    end
    tensortrace(A,labelsA,labelsC;basesize=basesize)
end

# In place method
#-----------------
macro gentracekernel(N1,N2,order,alpha,A,beta,C,startA,startC,odims,cdims,ostridesA,cstridesA,ostridesC)
    _gentracekernel(N1,N2,order,alpha,A,beta,C,startA,startC,odims,cdims,ostridesA,cstridesA,ostridesC)
end
function _gentracekernel(N1::Int,N2::Int,order::Symbol,alpha::Symbol,A::Symbol,beta::Symbol,C::Symbol,
    startA::Symbol,startC::Symbol,odims::Symbol,cdims::Symbol,ostridesA::Symbol,cstridesA::Symbol,ostridesC::Symbol)
    ex=quote
        local indA1, indA2, indC
        # we still have to implement other orders
        if $(esc(order))==0
            @stridedloops($N1, i, $(esc(cdims)), indA1, $(esc(startA)), $(esc(cstridesA)), begin
                local gamma
                gamma=beta
                @stridedloops($N2, j, $(esc(odims)), indA2, indA1, $(esc(ostridesA)), indC, $(esc(startA)), $(esc(ostridesC)), begin
                    @inbounds $(esc(C))[indC]=gamma*$(esc(C))[indC]+alpha*$(esc(A))[indA2]
                end)
                gamma=one(beta)
            end)
        else
            @stridedloops($N2, j, $(esc(odims)), indA1, $(esc(startA)), $(esc(ostridesA)), indC, $(esc(startA)), $(esc(ostridesC)), begin
                local localC
                localC=gamma*$(esc(C))[indC]
                @stridedloops($N1, i, $(esc(cdims)), indA2, indA1, $(esc(cstridesA)), @inbounds localC+=alpha*$(esc(A))[indA2])
                $(esc(C))[indC]=localC
            end)
        end
    end
    ex
end

const TRACEGENERATE=[(2,0),(3,1),(4,2),(4,0),(5,3),(5,1),(6,4),(6,2),(6,0)] 

@mngenerate NA NC typeof(C) TRACEGENERATE function tensortrace!{TA,NA,TC,NC}(alpha::Number,A::StridedArray{TA,NA},labelsA,beta::Number,C::StridedArray{TC,NC},labelsC,basesize::Int=1024)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
    NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC,basesize) # nothing to trace
    
    po=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))
    
    pc1=Array(Int,length(clabels))
    pc2=Array(Int,length(clabels))
    for i=1:length(clabels)
        pc1[i]=findfirst(labelsA,clabels[i])
        pc2[i]=findnext(labelsA,clabels[i],pc1[i]+1)
    end
    isperm(vcat(po,pc1,pc2)) || throw(LabelError("invalid label specification"))
    
    opensize=1
    for i = 1:N
        opensize*=size(C,i)
        size(A,po[i]) == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    tracesize=1
    for i = 1:div(NA-NC,2)
        tracesize*=size(A,pc2[i])
        size(A,pc1[i]) == size(A,pc2[i]) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    
    stridesA=collect(strides(A))
    cstridesA=stridesA[pc1]+stridesA[pc2]
    ostridesA=stridesA[po]
    p=sortperm(cstridesA)
    pc1=pc1[p]
    cstridesA=cstridesA[p]
    
    order=0
    if NC>0 && cstridesA[1] < minimum(ostridesA)
        order=1
    end
    
    @nexprs N d->(odims_{d} = size(C,d))
    @nexprs N d->(ostridesC_{d} = stride(C,d))
    @nexprs N d->(ostridesA_{d} = ostridesA[d])
    @nexprs N d->(minostrides_{d} = min(ostridesA_{d},ostridesC_{d}))
    
    @nexprs div(NA-NC,2) d->(cdims_{d} = size(A,pc1[d]))
    @nexprs div(NA-NC,2) d->(cstridesA_{d} = cstridesA[d])
    
    # initialize to zero
    if beta==zero(beta)
      fill!(C,zero(TC))
    end
    
    startA = 1
    local Alinear::Array{TA,NA}
    if isa(A, SubArray)
        startA = A.first_index
        Alinear = A.parent
    else
        Alinear = A
    end
    startC = 1
    local Clinear::Array{TC,NC}
    if isa(C, SubArray)
        startC = C.first_index
        Clinear = C.parent
    else
        Clinear = C
    end
    
    if opensize*(tracesize+1)<=8*PERMUTEBASELENGTH
      if order
        gamma=beta
        @nexprs 1 d->(cindA_{div(NA-NC,2)} = startA)
        @nloops(div(NA-NC,2), j, d->1:cdims_{d},
          d->(cindA_{d-1} = cindA_{d}),
          d->(cindA_{d} += cstridesA_{d}),
          begin
            @nexprs 1 d->(oindA_{N} = cindA_{0})
            @nexprs 1 d->(oindC_{N} = startC)
            @nloops(N, i, e->1:odims_{e},
              e->(oindA_{e-1} = oindA_{e};oindC_{e-1}=oindC_{e}),
              e->(oindA_{e} += ostridesA_{e};oindC_{e} += ostridesC_{e}),
              @inbounds C[oindC_0]=gamma*C[oindC_0]+alpha*A[oindA_0])
            gamma=one(beta)
          end)
        else
          @nexprs 1 d->(oindA_{N} = startA)
          @nexprs 1 d->(oindC_{N} = startC)
          @nloops(N, i, e->1:odims_{e},
            e->(oindA_{e-1} = oindA_{e};oindC_{e-1}=oindC_{e}),
            e->(oindA_{e} += ostridesA_{e};oindC_{e} += ostridesC_{e}),
            begin
              gamma=beta
              @nexprs 1 d->(cindA_{div(NA-NC,2)} = oindA_{0})
              @nloops(div(NA-NC,2), j, d->1:cdims_{d},
                d->(cindA_{d-1} = cindA_{d}),
                d->(cindA_{d} += cstridesA_{d}),
                begin
                  @inbounds C[oindC_0]=gamma*C[oindC_0]+alpha*A[cindA_0]
                  gamma=one(beta)
                end)
            end)
        end
    else
      # build recursive stack
      depth=iceil(log2(opensize/(OBASESIZE*OBASESIZE)))+iceil(log2(tracesize/CBASESIZE))
      level=1
      stackstep=zeros(Int,depth)
      @nexprs N d->begin
        stackobdims_{d} = zeros(Int,depth)
        stackobdims_{d}[level] = odims_{d}
      end
      @nexprs div(NA-NC,2) d->begin
        stackcbdims_{d} = zeros(Int,depth)
        stackcbdims_{d}[level] = cdims_{d}
      end
      stackoffsetA=zeros(Int,depth)
      stackoffsetA[level]=startA
      stackoffsetC=zeros(Int,depth)
      stackoffsetC[level]=startC
      stackdC=zeros(Int,depth)
      stackdA=zeros(Int,depth)
      stackbeta=zeros(typeof(beta),depth)
      stackbeta[level]=beta
      stackdmax=zeros(Int,depth)
      stackwhichd=zeros(Int,depth)
      stacknewdim=zeros(Int,depth)
      while level>0
        if level==depth # base case
          gamma0=stackbeta[depth]
          startA0=stackoffsetA[depth]
          startC0=stackoffsetC[depth]
          @nexprs N d->(obdims_{d} = stackobdims_{d}[depth])
          @nexprs div(NA-NC,2) d->(cbdims_{d} = stackcbdims_{d}[depth])
          if cstridesA_1 < minostrides_1
            gamma=gamm0
            @nexprs 1 d->(cindA_{div(NA-NC,2)} = startA0)
            @nloops(div(NA-NC,2), j, d->1:cdims_{d},
              d->(cindA_{d-1} = cindA_{d}),
              d->(cindA_{d} += cstridesA_{d}),
              begin
                @nexprs 1 d->(oindA_{N} = cindA_{0})
                @nexprs 1 d->(oindC_{N} = startC0)
                @nloops(N, i, e->1:obdims_{e},
                  e->(oindA_{e-1} = oindA_{e};oindC_{e-1}=oindC_{e}),
                  e->(oindA_{e} += ostridesA_{e};oindC_{e} += ostridesC_{e}),
                  @inbounds C[oindC_0]=gamma*C[oindC_0]+alpha*A[oindA_0])
                gamma=one(gamma0)
              end)
            else
              @nexprs 1 d->(oindA_{N} = startA0)
              @nexprs 1 d->(oindC_{N} = startC0)
              @nloops(N, i, e->1:obdims_{e},
                e->(oindA_{e-1} = oindA_{e};oindC_{e-1}=oindC_{e}),
                e->(oindA_{e} += ostridesA_{e};oindC_{e} += ostridesC_{e}),
                begin
                  gamma=gamma0
                  @nexprs 1 d->(cindA_{div(NA-NC,2)} = oindA_{0})
                  @nloops(div(NA-NC,2), j, d->1:cbdims_{d},
                    d->(cindA_{d-1} = cindA_{d}),
                    d->(cindA_{d} += cstridesA_{d}),
                    begin
                      @inbounds C[oindC_0]=gamma*C[oindC_0]+alpha*A[cindA_0]
                      gamma=one(gamma0)
                    end)
                end)
            end
          level-=1
        elseif stackstep[level]==0
          @nexprs N d->(obdims_{d} = stackobdims_{d}[level])
          @nexprs div(NA-NC,2) d->(cbdims_{d} = stackcbdims_{d}[level])
          # find which dimension to divide
          dmax=0
          whichd=0
          maxval=0
          newdim=0
          dC=0
          dA=0
          @nexprs N d->begin
            newmax=minostrides_{d}*obdims_{d}
            if obdims_{d}>1 && newmax>maxval
              dmax=d;
              whichd=0
              newdim=obdims_{d}>>1
              dC=ostridesC_{d}
              dA=ostridesA_{d}
              maxval=newmax
            end
          end
          @nexprs div(NA-NC,2) d->begin
            newmax=cstridesA_{d}*cbdims_{d}
            if cbdims_{d}>1 && newmax>maxval
              dmax=d
              whichd=1
              newdim=cbdims_{d}>>1
              dC=0
              dA=cstridesA_{d}
              maxval=newmax
            end
          end
          stacknewdim[level]=newdim
          stackdmax[level]=dmax
          stackwhichd[level]=whichd
          stackdC[level]=dC
          stackdA[level]=dA

          if whichd==0
            @nexprs N d->(stackobdims_{d}[level+1] = (d==dmax? newdim : obdims_{d}))
            @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = cbdims_{d})
          else
            @nexprs N d->(stackobdims_{d}[level+1] = obdims_{d})
            @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax ? newdim : cbdims_{d}))
          end
          stackoffsetC[level+1]=stackoffsetC[level]
          stackoffsetA[level+1]=stackoffsetA[level]
          stackbeta[level+1]=stackbeta[level]
          stackstep[level+1]=0

          stackstep[level]+=1
          level+=1
        elseif stackstep[level]==1
          @nexprs N d->(obdims_{d} = stackobdims_{d}[level])
          @nexprs div(NA-NC,2) d->(cbdims_{d} = stackcbdims_{d}[level])
          dmax=stackdmax[level]
          whichd=stackwhichd[level]
          newdim=stacknewdim[level]
          dC=stackdC[level]
          dA=stackdA[level]
          
          if whichd==0
            @nexprs N d->(stackobdims_{d}[level+1] = (d==dmax ? obdims_{d}-newdim : obdims_{d}))
            @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = cbdims_{d})
            stackoffsetC[level+1]=stackoffsetC[level]+dC*newdim
            stackoffsetA[level+1]=stackoffsetA[level]+dA*newdim
            stackbeta[level+1]=stackbeta[level]
          else
            @nexprs N d->(stackobdims_{d}[level+1] = obdims_{d})
            @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax ? cbdims_{d}-newdim : cbdims_{d}))
            stackoffsetC[level+1]=stackoffsetC[level]
            stackoffsetA[level+1]=stackoffsetA[level]+dA*newdim
            stackbeta[level+1]=one(beta)
          end
          stackstep[level+1]=0

          stackstep[level]+=1
          level+=1
        else
          level-=1
        end
      end
    end
    return C
end