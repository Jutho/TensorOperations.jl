# tensorcopy.jl
#
# Method for copying one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data.

# Simple method
# --------------
function tensorcopy(A,labelsA,outputlabels=labelsA)
    dims=size(A)
    perm=indexin(outputlabels,labelsA)
    length(perm) == ndims(A) || throw(LabelError("invalid label specification"))
    isperm(perm) || throw(LabelError("invalid label specification"))
    C=similar(A,dims[perm])
    tensorcopy!(A,labelsA,C,outputlabels)
    return C
end

# In-place method
#-----------------
@ngenerate N typeof(C) [1,2,3,4,5,6,7,8] function tensorcopy!{TA,TC,N}(A::StridedArray{TA,N},labelsA,C::StridedArray{TC,N},labelsC,basesize::Int=1024)
    iperm=indexin(labelsC,labelsA)
    length(iperm) == N || error("expected permutation of size $N, but length(perm)=$(length(perm))")
    isperm(iperm) || error("input is not a permutation")
    for i = 1:N
        size(A,iperm[i]) == size(C,i) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end
    N==0 && (C[1]=A[1]; return C)
    if iperm==collect(1:N)
        copy!(C,A)
    end
      
    @nexprs N d->(stridesA_{d} = stride(A,iperm[d]))
    @nexprs N d->(stridesC_{d} = stride(C,d))
    @nexprs N d->(dims_{d} = size(C,d))
    
    startA = 1
    local Alinear::Array{TA,N}
    if isa(A, SubArray)
        startA = A.first_index
        Alinear = A.parent
    else
        Alinear = A
    end
    startC = 1
    local Clinear::Array{TC,N}
    if isa(C, SubArray)
        startC = C.first_index
        Clinear = C.parent
    else
        Clinear = C
    end

    if length(C)<=4*PERMUTEBASELENGTH
        @stridedloops(N, i, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds Clinear[indC]=Alinear[indA])
    else
        @nexprs N d->(minstrides_{d} = min(stridesA_{d},stridesC_{d}))

        # build recursive stack
        depth=iceil(log2(length(C)/PERMUTEBASELENGTH))+2 # 2 levels safety margin
        level=1 # level of recursion
        stackstep=zeros(Int,depth) # record step of algorithm at the different recursion level
        stackstep[level]=0
        stackblength=zeros(Int,depth)
        stackblength[level]=length(C)
        @nexprs N d->begin
            stackbdims_{d} = zeros(Int,depth)
            stackbdims_{d}[level] = dims_{d}
        end
        stackbstartA=zeros(Int,depth)
        stackbstartA[level]=startA
        stackbstartC=zeros(Int,depth)
        stackbstartC[level]=startC
        
        stackdC=zeros(Int,depth)
        stackdA=zeros(Int,depth)
        stackdmax=zeros(Int,depth)
        stacknewdim=zeros(Int,depth)
        stackolddim=zeros(Int,depth)
        
        while level>0
            step=stackstep[level]
            blength=stackblength[level]
            @nexprs N d->(bdims_{d} = stackbdims_{d}[level])
            bstartA=stackbstartA[level]
            bstartC=stackbstartC[level]
            
            if blength<=PERMUTEBASELENGTH || level==depth # base case
                @stridedloops(N, i, bdims, indA, bstartA, stridesA, indC, bstartC, stridesC, @inbounds Clinear[indC]=Alinear[indA])
                level-=1
            elseif step==0
                # find which dimension to divide
                dmax=0
                maxval=0
                newdim=0
                olddim=0
                dC=0
                dA=0
                @nexprs N d->begin
                    newmax=bdims_{d}*minstrides_{d}
                    if bdims_{d}>1 && newmax>maxval
                        dmax=d
                        olddim=bdims_{d}
                        newdim=olddim>>1
                        dC=stridesC_{d}
                        dA=stridesA_{d}
                        maxval=newmax
                    end
                end
                stackolddim[level]=olddim
                stacknewdim[level]=newdim
                stackdmax[level]=dmax
                stackdC[level]=dC
                stackdA[level]=dA
                
                stackstep[level+1]=0
                @nexprs N d->(stackbdims_{d}[level+1] = (d==dmax ? newdim : bdims_{d}))
                stackblength[level+1]=div(blength,olddim)*newdim
                stackbstartA[level+1]=bstartA
                stackbstartC[level+1]=bstartC

                stackstep[level]+=1
                level+=1
            elseif step==1
                olddim=stackolddim[level]
                newdim=stacknewdim[level]
                dmax=stackdmax[level]
                dC=stackdC[level]
                dA=stackdA[level]

                stackstep[level+1]=0
                @nexprs N d->(stackbdims_{d}[level+1] = (d==dmax ? olddim-newdim : bdims_{d}))
                stackblength[level+1]=div(blength,olddim)*(olddim-newdim)
                stackbstartA[level+1]=bstartA+dA*newdim
                stackbstartC[level+1]=bstartC+dC*newdim

                stackstep[level]+=1
                level+=1
            else
                level-=1
            end
        end
    end
    return C
end