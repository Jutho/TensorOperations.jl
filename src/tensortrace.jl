# tensorcopy.jl
#
# Method for tracing some of the indices of a tensor and
# adding the result to another tensor.

# Simple method
#---------------
function tensortrace(A::StridedArray,labelsA,outputlabels)
    dimsA=size(A)
    C=similar(A,dimsA[indexin(outputlabels,labelsA)])
    tensortrace!(1,A,labelsA,0,C,outputlabels)
end
function tensortrace(A::StridedArray,labelsA) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,0)
    sizehint(labelsC,length(ulabelsA))
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            push!(labelsC,ulabelsA[j])
        end
    end
    tensortrace(A,labelsA,labelsC)
end


function tensortrace!(alpha::Number,A::StridedArray,labelsA,beta::Number,C::StridedArray,labelsC)
    NA=ndims(A)
    NC=ndims(C)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
    NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC) # nothing to trace

    oindA=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))

    cindA1=Array(Int,length(clabels))
    cindA2=Array(Int,length(clabels))
    for i=1:length(clabels)
        cindA1[i]=findfirst(labelsA,clabels[i])
        cindA2[i]=findnext(labelsA,clabels[i],cindA1[i]+1)
    end
    isperm(vcat(oindA,cindA1,cindA2)) || throw(LabelError("invalid label specification"))

    for i = 1:NC
        size(A,oindA[i]) == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    for i = 1:div(NA-NC,2)
        size(A,cindA1[i]) == size(A,cindA2[i]) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    tensortrace_native!(alpha,A,beta,C,oindA,cindA1,cindA2)
end


# In place method
#-----------------
const TRACEGENERATE=[(2,0),(3,1),(4,2),(4,0),(5,3),(5,1),(6,4),(6,2),(6,0)]

@eval @ngenerate (NA,NC) typeof(C) $TRACEGENERATE function tensortrace_native!{TA,NA,TC,NC}(alpha::Number,A::StridedArray{TA,NA},beta::Number,C::StridedArray{TC,NC},oindA,cindA1,cindA2)
    stridesA=collect(strides(A))
    cstridesA=stridesA[cindA1]+stridesA[cindA2]
    ostridesA=stridesA[oindA]
    p=sortperm(cstridesA)
    cindA1=cindA1[p]
    cstridesA=cstridesA[p]
    
    order=0
    if NC==0 || cstridesA[1] < minimum(ostridesA)
        order=1
    end
    
    olength=1
    @nexprs NC d->(odims_{d} = size(C,d);olength*=odims_{d})
    @nexprs NC d->(ostridesC_{d} = stride(C,d))
    @nexprs NC d->(ostridesA_{d} = ostridesA[d])
    
    clength=1
    @nexprs div(NA-NC,2) d->(cdims_{d} = size(A,cindA1[d]);clength*=cdims_{d})
    @nexprs div(NA-NC,2) d->(cstridesA_{d} = cstridesA[d])
    
    # initialize to zero
    if beta==0
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
    
    if olength*(clength+1)<=8*TBASELENGTH
        @gentracekernel(div(NA-NC,2),NC,order,alpha,Alinear,beta,Clinear,startA,startC,odims,cdims,ostridesA,cstridesA,ostridesC)
    else
        @nexprs NC d->(minostrides_{d} = min(ostridesA_{d},ostridesC_{d}))

        # build recursive stack
        depth=iceil(log2(olength*(clength+1)/2/TBASELENGTH))+2 # 2 levels safety margin
        level=1 # level of recursion
        stackpos=zeros(Int,depth) # record position of algorithm at the different recursion level
        stackpos[level]=0
        stackoblength=zeros(Int,depth)
        stackoblength[level]=olength
        stackcblength=zeros(Int,depth)
        stackcblength[level]=clength
        @nexprs NC d->begin
            stackobdims_{d} = zeros(Int,depth)
            stackobdims_{d}[level] = odims_{d}
        end
        @nexprs div(NA-NC,2) d->begin
            stackcbdims_{d} = zeros(Int,depth)
            stackcbdims_{d}[level] = cdims_{d}
        end
        stackbstartA=zeros(Int,depth)
        stackbstartA[level]=startA
        stackbstartC=zeros(Int,depth)
        stackbstartC[level]=startC
        stackgamma=zeros(typeof(beta),depth)
        stackgamma[level]=beta

        stackdC=zeros(Int,depth)
        stackdA=zeros(Int,depth)
        stackdmax=zeros(Int,depth)
        stackwhichd=zeros(Int,depth)
        stacknewdim=zeros(Int,depth)
        stackolddim=zeros(Int,depth)

        while level>0
            pos=stackpos[level]
            oblength=stackoblength[level]
            cblength=stackcblength[level]
            @nexprs NC d->(obdims_{d} = stackobdims_{d}[level])
            @nexprs div(NA-NC,2) d->(cbdims_{d} = stackcbdims_{d}[level])
            bstartA=stackbstartA[level]
            bstartC=stackbstartC[level]
            gamma=stackgamma[level]

            if oblength*(cblength+1)<=2*TBASELENGTH || level==depth # base case
                @gentracekernel(div(NA-NC,2),NC,order,alpha,Alinear,gamma,Clinear,bstartA,bstartC,obdims,cbdims,ostridesA,cstridesA,ostridesC)
                level-=1
            elseif pos==0
                # find which dimension to divide
                dmax=0
                whichd=0
                maxval=0
                newdim=0
                olddim=0
                dC=0
                dA=0
                @nexprs NC d->begin
                    newmax=obdims_{d}*minostrides_{d}
                    if obdims_{d}>1 && newmax>maxval
                        dmax=d
                        whichd=1
                        olddim=obdims_{d}
                        newdim=olddim>>1
                        dC=ostridesC_{d}
                        dA=ostridesA_{d}
                        maxval=newmax
                    end
                end
                @nexprs div(NA-NC,2) d->begin
                    newmax=cbdims_{d}*cstridesA_{d}
                    if cbdims_{d}>1 && newmax>maxval
                        dmax=d
                        whichd=2
                        olddim=cbdims_{d}
                        newdim=olddim>>1
                        dC=0
                        dA=cstridesA_{d}
                        maxval=newmax
                    end
                end
                stackolddim[level]=olddim
                stacknewdim[level]=newdim
                stackdmax[level]=dmax
                stackwhichd[level]=whichd
                stackdC[level]=dC
                stackdA[level]=dA

                stackpos[level+1]=0
                @nexprs NC d->(stackobdims_{d}[level+1] = (d==dmax && whichd==1 ? newdim : obdims_{d}))
                @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax && whichd==2 ? newdim : cbdims_{d}))
                stackoblength[level+1]=whichd==1 ? div(oblength,olddim)*newdim : oblength
                stackcblength[level+1]=whichd==2 ? div(cblength,olddim)*newdim : cblength
                stackbstartA[level+1]=bstartA
                stackbstartC[level+1]=bstartC
                stackgamma[level+1]=gamma

                stackpos[level]+=1
                level+=1
            elseif pos==1
                olddim=stackolddim[level]
                newdim=stacknewdim[level]
                dmax=stackdmax[level]
                whichd=stackwhichd[level]
                dC=stackdC[level]
                dA=stackdA[level]

                stackpos[level+1]=0
                @nexprs NC d->(stackobdims_{d}[level+1] = (d==dmax && whichd==1 ? olddim-newdim : obdims_{d}))
                @nexprs div(NA-NC,2) d->(stackcbdims_{d}[level+1] = (d==dmax && whichd==2 ? olddim-newdim : cbdims_{d}))
                stackoblength[level+1]=whichd==1 ? div(oblength,olddim)*(olddim-newdim) : oblength
                stackcblength[level+1]=whichd==2 ? div(cblength,olddim)*(olddim-newdim) : cblength
                stackbstartA[level+1]=bstartA+newdim*dA
                stackbstartC[level+1]=bstartC+newdim*dC
                stackgamma[level+1]=whichd==2 ? one(gamma) : gamma

                stackpos[level]+=1
                level+=1
            else
                level-=1
            end
        end
    end
    return C
end