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
    return tensorcopy!(A,labelsA,C,outputlabels)
end

# In-place method
#-----------------
function tensorcopy!{T1,T2,N}(A::StridedArray{T1,N},labelsA,C::StridedArray{T2,N},labelsC)
    dims=size(A)
    perm=indexin(labelsA,labelsC)
    if perm==collect(1:N)
        copy!(C,A)
    else
        length(perm) == N || throw(LabelError("invalid label specification"))
        isperm(perm) || throw(LabelError("invalid label specification"))
        (isbits(T1) && isbits(T2)) || error("only arrays of bitstypes are supported")
        for i = 1:length(perm)
            dims[i] == size(C,perm[i]) || throw(DimensionMismatch("destination tensor of incorrect size"))
        end

        stridesC=strides(C)[perm]
        stridesA=strides(A)
    
        unsafe_tensorcopy!(dims,pointer(A),stridesA,pointer(C),stridesC)
    end
    return C
end

# Low-level method
#------------------
@ngenerate N Ptr{T} function unsafe_tensorcopy!{T,N}(dims::NTuple{N,Int},A::Ptr{T},stridesA::NTuple{N,Int},C::Ptr{T},stridesC::NTuple{N,Int},bdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = bdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_0)
            @nexprs 1 e->(ind2C_{N} = indC_0)
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,unsafe_load(A,ind2A_0),ind2C_0)) #BODY
        end)
    return C
end
@ngenerate N Ptr{T2} function unsafe_tensorcopy!{T1,T2,N}(dims::NTuple{N,Int},A::Ptr{T1},stridesA::NTuple{N,Int},C::Ptr{T2},stridesC::NTuple{N,Int},bdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = bdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_0)
            @nexprs 1 e->(ind2C_{N} = indC_0)
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,convert(T2,unsafe_load(A,ind2A_0)),ind2C_0)) #BODY
        end)
    return C
end
unsafe_tensorcopy!{T1,T2,N}(dims::NTuple{N,Int},A::Ptr{T1},stridesA::NTuple{N,Int},C::Ptr{T2},stridesC::NTuple{N,Int})=unsafe_tensorcopy!(dims,A,stridesA,C,stridesC,blockdims1(dims,sizeof(T1),stridesA,sizeof(T2),stridesC))
