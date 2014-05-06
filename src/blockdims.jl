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
    if N==0
        return ()
    else
        pA=sortperm(collect(stridesA))
        pB=sortperm(collect(stridesB))
    
        # determine cache
        cacheline=64
        effectivecachesize=25600 # 64*400 = ifloor(cachesize/1.28) with cachesize=32k and 1.28 safety margin to prevent complete cachefill
    
        # if smallest stride of A or B is not 1, then the effect size a subblock of A
        # or B will take in the cache depends not only on the element size but also on
        # the number of unused data that will be copied together with every element
        cachesizeA=min(elszA*stridesA[pA[1]],cacheline)
        cachesizeB=min(elszB*stridesB[pB[1]],cacheline)

        # check if complete data fits into cache:
        if (cachesizeA+cachesizeB)*prod(dims)<=effectivecachesize
            return dims
        end
        
        # cache friendly blocking strategy    
        bdims=ones(Int,N)
        i=1
        j=1
        # loop will try to make blocks maximal along dimensions of minimal strides
        # for both A and B, until the blockdim equals the full dim along those
        # dimensions, and then continue with the next dimensions
        while true
            while bdims[pA[i]]==dims[pA[i]]
                i+=1
            end
            bdims[pA[i]]+=1
            if (cachesizeA+cachesizeB)*prod(bdims)>effectivecachesize # this must become true at some point
                bdims[pA[i]]-=1
                break
            end
        
            while bdims[pB[j]]==dims[pB[j]]
                j+=1
            end
            bdims[pB[j]]+=1
            if (cachesizeA+cachesizeB)*prod(bdims)>effectivecachesize # this must become true at some point
                bdims[pB[j]]-=1
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
        pA1=sortperm(collect(stridesA1))
        pA2=sortperm(collect(stridesA2))
        pB=sortperm(collect(stridesB1))
        pC=sortperm(collect(stridesC2))
    
        # determine cache
        cacheline=64
        effectivecachesize=25600 # 64*400 = ifloor(cachesize/1.28) with cachesize=32k and 1.28 safety margin to prevent complete cachefill
    
        # if smallest stride of A or B is not 1, then the effect size a subblock of A
        # or B will take in the cache depends not only on the element size but also on
        # the number of unused data that will be copied together with every element
        cachesizeA=min(elszA*stridesA1[pA1[1]],elszA*stridesA2[pA2[1]],cacheline)
        cachesizeB=min(elszB*stridesB1[pB[1]],cacheline)
        cachesizeC=min(elszC*stridesC2[pC[1]],cacheline)

        # check if complete data fits in cache
        if cachesizeA*prod(dims1)*prod(dims2)+cachesizeB*prod(dims1)+cachesizeC*prod(dims2)<=effectivecachesize
            return dims1,dims2
        end
        
        # cache friendly blocking strategy:
        bdims1=ones(Int,N1)
        bdims2=ones(Int,N2)
        i1=1
        i2=1
        j1=1
        j2=1
        # loop will try to make blocks maximal along dimensions of minimal strides
        # for both A and B, until the blockdim equals the full dim along those
        # dimensions, and then continue with the next dimensions
        while true
            while i1<=N1 && bdims1[pA1[i1]]==dims1[pA1[i1]]
                i1+=1
            end
            while i2<=N2 && bdims2[pA2[i2]]==dims2[pA2[i2]]
                i2+=1
            end
            if i1<=N1 && (i2>N2 || stridesA1[pA1[i1]]<=stridesA2[pA2[i2]])
                bdims1[pA1[i1]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims1[pA1[i1]]-=1
                    break
                end
                
                while bdims1[pB[j1]]==dims1[pB[j1]]
                    j1+=1
                end
                bdims1[pB[j1]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims1[pB[j1]]-=1
                    break
                end
            elseif i2<=N2
                bdims2[pA2[i2]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims2[pA2[i2]]-=1
                    break
                end
                
                while bdims2[pC[j2]]==dims2[pC[j2]]
                    j2+=1
                end
                bdims2[pC[j2]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)+cachesizeC*prod(bdims2)>effectivecachesize
                    bdims[pC[j2]]-=1
                    break
                end
            else # should never happen
                warning("this should not happen")
                break
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
        pA1=sortperm(collect(stridesA1))
        pA2=sortperm(collect(stridesA2))
        pB1=sortperm(collect(stridesB1))
        pB3=sortperm(collect(stridesB3))
        pC2=sortperm(collect(stridesC2))
        pC3=sortperm(collect(stridesC3))
    
        # determine cache
        cacheline=64
        effectivecachesize=25600 # 64*400 = ifloor(cachesize/1.28) with cachesize=32k and 1.28 safety margin to prevent complete cachefill
    
        # if smallest stride of A or B is not 1, then the effect size a subblock of A
        # or B will take in the cache depends not only on the element size but also on
        # the number of unused data that will be copied together with every element
        cachesizeA=min(elszA*stridesA1[pA1[1]],elszA*stridesA2[pA2[1]],cacheline)
        cachesizeB=min(elszB*stridesB1[pB1[1]],elszB*stridesB3[pB3[1]],cacheline)
        cachesizeC=min(elszC*stridesC2[pC2[1]],elszC*stridesC3[pC3[1]],cacheline)
        
        # check if complete data fits in cache
        if cachesizeA*prod(dims1)*prod(dims2)+cachesizeB*prod(dims1)*prod(dims3)+cachesizeC*prod(dims2)*prod(dims3)<=effectivecachesize
            return dims1,dims2,dims3
        end
        
        # Cache-friendly blocking strategy:
        bdims1=ones(Int,N1)
        bdims2=ones(Int,N2)
        bdims3=ones(Int,N3)
        i1=1
        i2=1
        i3=1
        j1=1
        j2=1
        j3=1
        while true
            while i1<=N1 && bdims1[pB1[i1]]==dims1[pB1[i1]]
                i1+=1
            end
            while i2<=N2 && bdims2[pC2[i2]]==dims2[pC2[i2]]
                i2+=1
            end
            while i3<=N3 && bdims3[pB3[i3]]==dims2[pB3[i3]]
                i3+=1
            end
            
            if i1<=N1
                bdims1[pB1[i1]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims1[pB1[i1]]-=1
                    break
                end
                while j1<=N1 && bdims1[pA1[j1]]==dims1[pA1[j1]]
                    j1+=1
                end
                if j1<=N1
                    bdims1[pA1[j1]]+=1
                    if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                        bdims1[pA1[j1]]-=1
                        break
                    end
                end
            end
            if i2<=N2
                bdims2[pC2[i2]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims2[pC2[i2]]-=1
                    break
                end
                while j2<=N2 && bdims2[pA2[j2]]==dims2[pA2[j2]]
                    j2+=1
                end
                if j2<=N2
                    bdims2[pA2[j2]]+=1
                    if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                        bdims2[pA2[j2]]-=1
                        break
                    end
                end
            end
            if i3<=N3
                bdims3[pB3[i3]]+=1
                if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                    bdims3[pB3[i3]]-=1
                    break
                end
                while j3<=N3 && bdims3[pC3[j3]]==dims3[pC3[j3]]
                    j3+=1
                end
                if j3<=N3
                    bdims3[pC3[j3]]+=1
                    if cachesizeA*prod(bdims1)*prod(bdims2)+cachesizeB*prod(bdims1)*prod(bdims3)+cachesizeC*prod(bdims2)*prod(bdims3)>effectivecachesize
                        bdims3[pC3[j3]]-=1
                        break
                    end
                end
            end
        end
        return tuple(bdims1...),tuple(bdims2...),tuple(bdims3...)
    end
end