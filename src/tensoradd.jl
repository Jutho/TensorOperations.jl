# tensoradd.jl
#
# Method for adding one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data.

# Simple method
# --------------
function tensoradd(A,labelsA,B,labelsB,outputlabels=labelsA)
    dims=size(A)
    T=promote_type(eltype(A),eltype(B))
    perm=indexin(outputlabels,labelsA)
    length(perm) == ndims(A) || throw(LabelError("invalid label specification"))
    isperm(perm) || throw(LabelError("invalid label specification"))
    C=similar(A,T,dims[perm])
    tensorcopy!(A,labelsA,C,outputlabels)
    return tensoradd!(one(T),B,labelsB,one(T),C,outputlabels)
end

# In-place method
#-----------------
function tensoradd!{T1,T2,N}(alpha::Number,A::StridedArray{T1,N},labelsA,beta::Number,C::StridedArray{T2,N},labelsC)
    dims=size(A)
    perm=indexin(labelsA,labelsC)

    length(perm) == N || throw(LabelError("invalid label specification"))
    isperm(perm) || throw(LabelError("invalid label specification"))
    (isbits(T1) && isbits(T2)) || error("only arrays of bitstypes are supported")
    for i = 1:length(perm)
        dims[i] == size(C,perm[i]) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end

    stridesC=strides(C)[perm]
    stridesA=strides(A)

    unsafe_tensoradd!(dims,convert(T2,alpha),pointer(A),stridesA,convert(T2,beta),pointer(C),stridesC)

    return C
end

# Low-level method
#------------------
# TENSORADD
@ngenerate N Ptr{T} function unsafe_tensoradd!{T,N}(dims::NTuple{N,Int},alpha::T,A::Ptr{T},stridesA::NTuple{N,Int},beta::T,C::Ptr{T},stridesC::NTuple{N,Int},bdims::NTuple{N,Int})
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
                unsafe_store!(C,beta*unsafe_load(C,ind2C_0)+alpha*unsafe_load(A,ind2A_0),ind2C_0)) #BODY
        end)
    return C
end
@ngenerate N Ptr{T2} function unsafe_tensoradd!{T1,T2,N}(dims::NTuple{N,Int},alpha::T2,A::Ptr{T1},stridesA::NTuple{N,Int},beta::T2,C::Ptr{T2},stridesC::NTuple{N,Int},bdims::NTuple{N,Int})
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
                unsafe_store!(C,beta*unsafe_load(C,ind2C_0)+alpha*convert(T2,unsafe_load(A,ind2A_0)),ind2C_0)) #BODY
        end)
    return C
end
unsafe_tensoradd!{T1,T2,N}(dims::NTuple{N,Int},alpha::T2,A::Ptr{T1},stridesA::NTuple{N,Int},beta::T2,C::Ptr{T2},stridesC::NTuple{N,Int})=unsafe_tensoradd!(dims,alpha,A,stridesA,beta,C,stridesC,blockdims1(dims,sizeof(T1),stridesA,sizeof(T2),stridesC))
