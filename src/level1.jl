# LEVEL 1: no contraction or full contraction

function tensorcopy!{T1,T2,N}(C::StridedArray{T1,N},labelsC,A::StridedArray{T2,N},labelsA)
    dims=size(A)
    perm=indexin(labelsA,labelsC)
    if perm==collect(1:N)
        copy!(C,A)
    else
        length(perm) == N || error("invalid label specification")
        isperm(perm) || error("invalid label specification")
        (isbits(T1) && isbits(T2)) || error("only arrays of bitstypes are supported")
        for i = 1:length(perm)
            dims[i] == size(C,perm[i]) || throw(DimensionMismatch("destination tensor of incorrect size"))
        end

        stridesC=strides(C)[perm]
        stridesA=strides(A)
    
        unsafe_tensorcopy!(pointer(C),pointer(A),stridesC,stridesA,dims)
    end
    return C
end
function tensorcopy{T,N}(labelsC,A::StridedArray{T,N},labelsA)
    dims=size(A)
    perm=indexin(labelsC,labelsA)
    C=similar(A,dims[invperm(perm)])
    return tensorcopy!(C,labelsC,A,labelsA)
end

function tensoradd!{T1,T2,N}(beta::Number,C::StridedArray{T1,N},labelsC,alpha::Number,A::StridedArray{T2,N},labelsA)
    dims=size(A)
    perm=indexin(labelsA,labelsC)

    length(perm) == N || error("invalid label specification")
    isperm(perm) || error("invalid label specification")
    (isbits(T1) && isbits(T2)) || error("only arrays of bitstypes are supported")
    for i = 1:length(perm)
        dims[i] == size(C,perm[i]) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end

    stridesC=strides(C)[perm]
    stridesA=strides(A)

    unsafe_tensoradd!(convert(T1,beta),pointer(C),convert(T1,alpha),pointer(A),stridesC,stridesA,dims)

    return C
end
function tensoradd{T1,T2,N}(A::StridedArray{T1,N},labelsA,B::StridedArray{T2,N},labelsB)
    dims=size(A)
    T=promote_type(T1,T2)
    C=similar(A,T)
    copy!(C,A)
    return tensoradd!(one(T),C,labelsA,one(T),B,labelsB)
end

function tensordot{T1,T2,N}(A::StridedArray{T1,N},labelsA,B::StridedArray{T2,N},labelsB)
    dims=size(B)
    perm=indexin(labelsB,labelsA)

    length(perm) == N || error("invalid label specification")
    isperm(perm) || error("invalid label specification")
    (isbits(T1) && isbits(T2)) || error("only arrays of bitstypes are supported")
    for i = 1:length(perm)
        dims[i] == size(A,perm[i]) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end

    stridesA=strides(A)[perm]
    stridesB=strides(B)

    return unsafe_tensordot(pointer(A),pointer(B),stridesA,stridesB,dims)
end

# TENSORCOPY
@ngenerate N Ptr{T} function unsafe_tensorcopy!{T,N}(C::Ptr{T},A::Ptr{T},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int},blockdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = blockdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_{1})
            @nexprs 1 e->(ind2C_{N} = indC_{1})
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,unsafe_load(A,ind2A_1),ind2C_1)) #BODY
        end)
    return C
end
@ngenerate N Ptr{T1} function unsafe_tensorcopy!{T1,T2,N}(C::Ptr{T1},A::Ptr{T2},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int},blockdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = blockdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_{1})
            @nexprs 1 e->(ind2C_{N} = indC_{1})
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,convert(T1,unsafe_load(A,ind2A_1)),ind2C_1)) #BODY
        end)
    return C
end
unsafe_tensorcopy!{T1,T2,N}(C::Ptr{T1},A::Ptr{T2},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int})=unsafe_tensorcopy!(C,A,stridesC,stridesA,dims,level1blockdims(dims,sizeof(T1),sizeof(T2),stridesC,stridesA))

# TENSORADD
@ngenerate N Ptr{T} function unsafe_tensoradd!{T,N}(beta::T,C::Ptr{T},alpha::T,A::Ptr{T},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int},blockdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = blockdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_{1})
            @nexprs 1 e->(ind2C_{N} = indC_{1})
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,beta*unsafe_load(C,ind2C_1)+alpha*unsafe_load(A,ind2A_1),ind2C_1)) #BODY
        end)
    return C
end
@ngenerate N Ptr{T1} function unsafe_tensoradd!{T1,T2,N}(beta::T1,C::Ptr{T1},alpha::T1,A::Ptr{T2},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int},blockdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = blockdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesC_{d} = stridesC[d])
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indC_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indC_{d} += bdims_{d}*stridesC_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_{1})
            @nexprs 1 e->(ind2C_{N} = indC_{1})
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2C_{e-1}=ind2C_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2C_{e} += stridesC_{e}), # POST
                unsafe_store!(C,beta*unsafe_load(C,ind2C_1)+alpha*convert(T1,unsafe_load(A,ind2A_1)),ind2C_1)) #BODY
        end)
    return C
end
unsafe_tensoradd!{T1,T2,N}(beta::T1,C::Ptr{T1},alpha::T1,A::Ptr{T2},stridesC::NTuple{N,Int},stridesA::NTuple{N,Int},dims::NTuple{N,Int})=unsafe_tensoradd!(beta,C,alpha,A,stridesC,stridesA,dims,level1blockdims(dims,sizeof(T1),sizeof(T2),stridesC,stridesA))

# TENSORDOT
@ngenerate N promote_type(T1,T2) function unsafe_tensordot{T1,T2,N}(A::Ptr{T1},B::Ptr{T2},stridesA::NTuple{N,Int},stridesB::NTuple{N,Int},dims::NTuple{N,Int},blockdims::NTuple{N,Int})
    # calculate dims as variables
    @nexprs N d->(dims_{d} = dims[d])
    @nexprs N d->(bdims_{d} = blockdims[d])
    # calculate strides as variables
    @nexprs N d->(stridesA_{d} = stridesA[d])
    @nexprs N d->(stridesB_{d} = stridesB[d])
    
    T=promote_type(T1,T2)
    C=zero(T)
    
    @nexprs 1 d->(indA_{N} = 1)
    @nexprs 1 d->(indB_{N} = 1)
    @nloops(N, outer, d->1:bdims_{d}:dims_{d},
        d->(indA_{d-1} = indA_{d};indB_{d-1}=indB_{d}), # PRE
        d->(indA_{d} += bdims_{d}*stridesA_{d};indB_{d} += bdims_{d}*stridesB_{d}), # POST
        begin # BODY
            @nexprs 1 e->(ind2A_{N} = indA_{1})
            @nexprs 1 e->(ind2B_{N} = indB_{1})
            @nloops(N, inner, e->outer_{e}:min(outer_{e}+bdims_{e}-1,dims_{e}),
                e->(ind2A_{e-1} = ind2A_{e};ind2B_{e-1}=ind2B_{e}), # PRE
                e->(ind2A_{e} += stridesA_{e};ind2B_{e} += stridesB_{e}), # POST
                C+=unsafe_load(A,ind2A_1)*unsafe_load(B,ind2B_1)) #BODY
        end)
    return C
end
unsafe_tensordot{T1,T2,N}(A::Ptr{T1},B::Ptr{T2},stridesA::NTuple{N,Int},stridesB::NTuple{N,Int},dims::NTuple{N,Int})=unsafe_tensordot(A,B,stridesA,stridesB,dims,level1blockdims(dims,sizeof(T1),sizeof(T2),stridesA,stridesB))

# AUXILIARY FUNCTION
function level1blockdims{N}(dims::NTuple{N,Int},elsz1::Int,elsz2::Int,strides1::NTuple{Int},strides2::NTuple{Int})
    # Try to compute an cache-friendly blocking strategy for a set of
    # dimensions dims, that are shared by two arrays (either for contraction or
    # open), where array 1 and 2 contain elements with element sizes elsz1 and
    # elsz2 respectively, and have strides along the corrent dimensions
    # contained in strides1 and strides2.

    # Cache-friendly blocking strategy:
    effectivecachesize=ifloor(cachesize/1.2) # 1.2 safety margin
    blockstep=ones(Int,N)
    for i=1:N
        blockstep[i]=max(1,div(cacheline,elsz1*strides1[i]),div(cacheline,elsz2*strides2[i]))
        # blockstep is the number of elements along that dimension that can be expected to be
        # within a single cacheline for either array 1 or 2
    end
    blockdims=zeros(Int,N)
    cachesize1=elsz1*minimum(strides1)
    cachesize2=elsz2*minimum(strides2)
    while true
        i=indmin(blockdims)
        blockdims[i]+=blockstep[i]
        if (cachesize1+cachesize2)*prod(blockdims)>effectivecachesize
            blockdims[i]-=blockstep[i]
            break
        end
    end
    return tuple(blockdims...)
end