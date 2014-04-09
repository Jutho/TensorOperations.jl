# blockdims.jl
#
# Contains auxiliary functions for computing a cache-friendly blockings strategy for the various
# tensor operations in this package.

function blockdims1{N}(dims::NTuple{N,Int},elszA::Int,stridesA::NTuple{N,Int},elszB::Int,stridesB::NTuple{N,Int})
    # Try to compute a cache-friendly blocking strategy for a set of
    # dimensions dims, that are shared by two arrays A and B (either for contraction or
    # open), where array A and B contain elements with element sizes elszA and
    # elszB respectively, and have strides along the corrent dimensions
    # contained in stridesA and stridesB.
    
    # special case
    if N==0
        return ()
    else
        # determine cache
        effectivecachesize=ifloor(cachesize/1.2) # 1.2 safety margin
        cachesizeA=elszA*minimum(stridesA)
        cachesizeB=elszB*minimum(stridesB)
        
        # check if complete data fits into cache:
        if (cachesizeA+cachesizeB)*prod(dims)<=effectivecachesize
            return dims
        end
        
        # cache-friendly blocking strategy:
        bstep=ones(Int,N)
        for i=1:N
            bstep[i]=max(1,div(cacheline,elszA*stridesA[i]),div(cacheline,elszB*stridesB[i]))
            # bstep is the number of elements along that dimension that can be expected to be
            # within a single cacheline for either array 1 or 2
        end
        bdims=zeros(Int,N)
        bdimscopy=zeros(Int,N) # copy where entries will be put very large once they are larger then corresponding dims
        while true
            i=indmin(bdimscopy)
            bdims[i]+=bstep[i]
            bdimscopy[i]+=bstep[i]
            if bdimscopy[i]>=dims[i]
                bdims[i]=dims[i]
                bdimscopy[i]=typemax(Int)
            end
            if (cachesizeA+cachesizeB)*prod(bdims)>effectivecachesize # this must become true at some point
                bdims[i]-=bstep[i]
                break
            end
        end
        return tuple(bdims...)
    end
end
function blockdims2{N1,N2}(dims1::NTuple{N1,Int},dims2::NTuple{N2,Int},elszA::Int,stridesA1::NTuple{N1,Int},stridesA2::NTuple{N2,Int},elszB::Int,stridesB1::NTuple{N1,Int},elszC::Int,stridesC2::NTuple{N2,Int})
    # Try to compute a cache-friendly blocking strategy for two sets of
    # dimensions dims1 and dims2, where dims1 are the dimensions of a set
    # of indices shared between two arrays A and B, and dims2 are the
    # the dimensions of a set of indices shared between A and a third
    # array C.
    
    if N1==0
        return (), blockdims1(dims2,elszA,stridesA2,elszC,stridesC2)
    elseif N2==0
        return blockdims1(dims1,elszA,stridesA1,elszB,stridesB1), ()
    else
        effectivecachesize=ifloor(cachesize/1.2) # 1.2 safety margin
        cachesizeA=elszA*min(minimum(stridesA1),minimum(stridesA2))
        cachesizeB=elszB*minimum(stridesB1)
        cachesizeC=elszC*minimum(stridesC2)
        if cachesizeA*prod(dims1)*prod(dims2)+cachesizeB*prod(dims1)+cachesizeC*prod(dims2)<=effectivecachesize
            return dims1,dims2
        end
        # cache-friendly blocking strategy:
        bstep1=ones(Int,N1)
        for i=1:N1
            bstep1[i]=max(1,div(cacheline,elszA*stridesA1[i]),div(cacheline,elszB*stridesB1[i]))
        end
        bstep2=ones(Int,N2)
        for i=1:N2
            bstep2[i]=max(1,div(cacheline,elszA*stridesA2[i]),div(cacheline,elszC*stridesC2[i]))
        end
        bdims1=zeros(Int,N1)
        bdims1copy=zeros(Int,N1)
        bdims2=zeros(Int,N2)
        bdims2copy=zeros(Int,N2)
        while true
            i=indmin(bdims1copy)
            j=indmin(bdims2copy)
            if bdims1copy[i]<=bdims2copy[j]
                bdims1[i]+=bstep1[i]
                bdims1copy[i]+=bstep1[i]
                if bdims1copy[i]>=dims1[i]
                    bdims1copy[i]=typemax(Int)
                end
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims1[i]-=bstep1[i]
                    break
                end
            else
                bdims2[j]+=bstep2[j]
                bdims2copy[j]+=bstep2[j]
                if bdims2copy[j]>=dims2[j]
                    bdims1copy[j]=typemax(Int)
                end
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims2[j]-=bstep2[j]
                    break
                end
            end
        end
        return tuple(bdims1...),tuple(bdims2...)
    end
end
function blockdims3{N1,N2,N3}(dims1::NTuple{N1,Int},dims2::NTuple{N2,Int},dims3::NTuple{N3,Int},elszA::Int,stridesA1::NTuple{N1,Int},stridesA2::NTuple{N2,Int},elszB::Int,stridesB1::NTuple{N1,Int},stridesB3::NTuple{N3,Int},elszC::Int,stridesC2::NTuple{N2,Int},stridesC3::NTuple{N3,Int})
    # Try to compute a cache-friendly blocking strategy for three sets of
    # dimensions dims1 and dims2, where dims1 are the dimensions of a set
    # of indices shared between two arrays A and B, dims2 are the
    # the dimensions of a set of indices shared between A and a third
    # array C, which also shares indices of dimensions dims3 with
    # array B. This is the typical case in general tensor contraction.
    
    if N1==0
        bdims2,bdims3=blockdims2(dims2,dims3,elszC,stridesC2,stridesC3,elszA,stridesA2,elszB,stridesB3)
        return (),bdims2,bdims3
    elseif N2==0
        bdims1,bdims3=blockdims2(dims1,dims3,elszB,stridesB1,stridesB3,elszA,stridesA1,elszC,stridesC3)
        return bdims1,(),bdims3
    elseif N3==0
        bdims1,bdims2=blockdims2(dims1,dims2,elszA,stridesA1,stridesA2,elszB,stridesB1,elszC,stridesC2)
        return bdims1,bdims2,()
    else
        # Cache-friendly blocking strategy:
        effectivecachesize=ifloor(cachesize/1.2) # 1.2 safety margin
        cachesizeA=elszA*min(minimum(stridesA1),minimum(stridesA2))
        cachesizeB=elszB*min(minimum(stridesB1),minimum(stridesB3))
        cachesizeC=elszC*min(minimum(stridesC2),minimum(stridesC3))
        if cachesizeA*prod(dims1)*prod(dims2)+cachesizeB*prod(dims1)*prod(dims3)+cachesizeC*prod(dims2)*prod(dims3)<=effectivecachesize
            return dims1,dims2,dims3
        end
        bstep1=ones(Int,N1)
        for i=1:N1
            bstep1[i]=max(1,div(cacheline,elszA*stridesA1[i]),div(cacheline,elszB*stridesB1[i]))
        end
        bstep2=ones(Int,N2)
        for i=1:N2
            bstep2[i]=max(1,div(cacheline,elszA*stridesA2[i]),div(cacheline,elszC*stridesC2[i]))
        end
        bstep3=ones(Int,N3)
        for i=1:N3
            bstep3[i]=max(1,div(cacheline,elszB*stridesB3[i]),div(cacheline,elszC*stridesC3[i]))
        end
        bdims1=zeros(Int,N1)
        bdims1copy=zeros(Int,N1)
        bdims2=zeros(Int,N2)
        bdims2copy=zeros(Int,N2)
        bdims3=zeros(Int,N3)
        bdims3copy=zeros(Int,N3)
        while true
            i=indmin(bdims1copy)
            j=indmin(bdims2copy)
            k=indmin(bdims3copy)
            if bdims1copy[i]<=bdims2copy[j] && bdims1copy[i]<=bdims3copy[k] # try to increase bdims1
                bdims1[i]+=bstep1[i]
                bdims1copy[i]+=bstep1[i]
                if bdims1copy[i]>=dims1[i]
                    bdims1copy[i]=typemax(Int)
                end
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims1[i]-=bstep1[i]
                    break
                end
            elseif bdims3copy[k]<=bdims2copy[j] # then try to increase bdims3
                bdims3[k]+=bstep3[k]
                bdims3copy[k]+=bstep3[k]
                if bdims3copy[k]>=dims1[k]
                    bdims3copy[k]=typemax(Int)
                end
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims3[k]-=bstep3[k]
                    break
                end
            else
                bdims2[j]+=bstep2[j]
                bdims2copy[j]+=bstep2[j]
                if bdims2copy[j]>=dims2[j]
                    bdims1copy[j]=typemax(Int)
                end
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims2[j]-=bstep2[j]
                    break
                end
            end
        end
        return tuple(bdims1...),tuple(bdims2...),tuple(bdims3...)
    end
end