# tensoradd.jl
#
# Method for adding one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data.

# Simple method
# --------------
function tensoradd(A::StridedArray,labelsA,B::StridedArray,labelsB,outputlabels=labelsA)
    dims=size(A)
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,dims[indexin(outputlabels,labelsA)])
    tensorcopy!(A,labelsA,C,outputlabels)
    tensoradd!(1,B,labelsB,1,C,outputlabels)
end

# In-place method
#-----------------
function tensoradd!(alpha::Number,A::StridedArray,labelsA,beta::Number,C::StridedArray,labelsC)
    NA=ndims(A)
    perm=indexin(labelsC,labelsA)
    length(perm) == NA || throw(LabelError("invalid label specification"))
    isperm(perm) || throw(LabelError("labels do not specify a valid permutation"))
    for i = 1:NA
        size(A,perm[i]) == size(C,i) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end
    NA==0 && (C[1]=beta*C[1]+alpha*A[1]; return C)
    perm==[1:NA] && return (beta==0 ? scale!(copy!(C,A),alpha) : Base.LinAlg.axpy!(alpha,A,scale!(C,beta)))
    beta==0 && return scale!(tensorcopy_native!(A,C,perm),alpha)
    tensoradd_native!(alpha,A,beta,C,perm)
end

# Implementation
#-----------------
@eval @ngenerate N typeof(C) $PERMUTEGENERATE function tensoradd_native!{TA,TC,N}(alpha::Number,A::StridedArray{TA,N},beta::Number,C::StridedArray{TC,N},perm)
    @nexprs N d->(stridesA_{d} = stride(A,perm[d]))
    @nexprs N d->(stridesC_{d} = stride(C,d))
    @nexprs N d->(dims_{d} = size(C,d))
  
    if beta==0
        fill!(C,zero(TC))
    end
    
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
  
    if length(C)<=4*TBASELENGTH
        @stridedloops(N, i, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds Clinear[indC]=beta*Clinear[indC]+alpha*Alinear[indA])
    else
        @nexprs N d->(minstrides_{d} = min(stridesA_{d},stridesC_{d}))

        # build recursive stack
        depth=iceil(log2(length(C)/TBASELENGTH))+2 # 2 levels safety margin
        level=1 # level of recursion
        stackpos=zeros(Int,depth) # record pos of algorithm at the different recursion level
        stackpos[level]=0
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
            pos=stackpos[level]
            blength=stackblength[level]
            @nexprs N d->(bdims_{d} = stackbdims_{d}[level])
            bstartA=stackbstartA[level]
            bstartC=stackbstartC[level]
            
            if blength<=TBASELENGTH || level==depth # base case
                @stridedloops(N, i, bdims, indA, bstartA, stridesA, indC, bstartC, stridesC, @inbounds Clinear[indC]=beta*Clinear[indC]+alpha*Alinear[indA])
                level-=1
            elseif pos==0
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
                
                stackpos[level+1]=0
                @nexprs N d->(stackbdims_{d}[level+1] = (d==dmax ? newdim : bdims_{d}))
                stackblength[level+1]=div(blength,olddim)*newdim
                stackbstartA[level+1]=bstartA
                stackbstartC[level+1]=bstartC

                stackpos[level]+=1
                level+=1
            elseif pos==1
                olddim=stackolddim[level]
                newdim=stacknewdim[level]
                dmax=stackdmax[level]
                dC=stackdC[level]
                dA=stackdA[level]

                stackpos[level+1]=0
                @nexprs N d->(stackbdims_{d}[level+1] = (d==dmax ? olddim-newdim : bdims_{d}))
                stackblength[level+1]=div(blength,olddim)*(olddim-newdim)
                stackbstartA[level+1]=bstartA+dA*newdim
                stackbstartC[level+1]=bstartC+dC*newdim

                stackpos[level]+=1
                level+=1
            else
                level-=1
            end
        end
    end
    return C
end