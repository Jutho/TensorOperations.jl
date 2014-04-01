# LEVEL 3: full tensor contraction: two sets of open indices and one set of contracted indices

function tensorcontract!{R,S,T}(beta::Number,C::StridedArray{R},labelsC,alpha::Number,A::StridedArray{S},labelsA,conjA::Char,B::StridedArray{T},labelsB,conjB::Char;method::Symbol=:BLAS)
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
    # -> method=:BLAS : permutes tensor (requires extra memory) and then calls
    #                   built-in (typically BLAS) multiplication
    # -> method=:native : calls multiplication on small subblocks of the tensor,
    #                     so that no new memory allocation is required
    
    # Get properties of input arrays
    NA=ndims(A)
    NB=ndims(B)
    NC=ndims(C)

    # Process labels, do some error checking and analyse problem structure
    #----------------------------------------------------------------------
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(IndexError("There should be a label for every index of the tensor"))
    end
    ulabelsA=unique(labelsA)
    ulabelsB=unique(labelsB)
    ulabelsC=unique(labelsC)
    if NA!=length(ulabelsA) || NB!=length(ulabelsB) || NC!=length(ulabelsC)
        throw(IndexError("There should be a unique label for every index of the tensor, handle inner contraction first with trace"))
    end

    clabels=intersect(ulabelsA,ulabelsB)
    numcontract=length(clabels)
    olabelsA=intersect(ulabelsA,ulabelsC)
    numopenA=length(olabelsA)
    olabelsB=intersect(ulabelsB,ulabelsC)
    numopenB=length(olabelsB)
    
    if numcontract+numopenA!=NA || numcontract+numopenB!=NB || numopenA+numopenB!=NC
        throw(IndexError("Invalid contraction pattern"))
    end
    
    # Check for genuine tensor contraction with open indices in both tensor A
    # and tensor B. If both numopenA and numopenB are zero, this corresponds to
    # a tensordot operation, where all indices are contracted and the result is
    # a scalar (corresponding to level 1 in BLAS terminonology). If one of them
    # is zero, this corresponds to a tensor projection, which would correspond
    # to a matrix vector multiplication (level 2 operation in BLAS terminology).
    if numopenA==0 && numopenB==0 # level 1
        s=tensordot(A,labelA,B,labelB)
        C[1]=beta*C[1]+alpha*s
        return C
    elseif numopenB==0 # level 2
        tensorprojection!(beta,C,labelC,alpha,A,conjA,labelA,B,labelB,conjB)
        return C
    elseif numopenA==0 # level 3
        tensorprojection!(beta,C,labelC,alpha,B,labelB,conjB,A,labelA,conjA)
        return C
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
        odimsA[i]==dimC[oindCA[i]] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    for i=1:numopenB
        odimsB[i]==dimC[oindCB[i]] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    
    # Perform contraction
    
    # The :BLAS method specification permutes A and B such that indopen and
    # indcontract are grouped, reshape them to matrices with all indopen on one
    # side and all indcontract on the other. Compute the data for C from
    # multiplying these matrices. Permute again to bring indices in requested
    # order.
    #
    # While this can potentially use the highly optimized BLAS matrix
    # multiplication, it needs three temporary arrays containing the
    # permuted copies of A, B and C. Memorywise this is far from optimal, and
    # the memory allocation and copying also impacts the computation time.
    
    if method==:BLAS
        # permute A
        if conjA=='C'
            Anew=similar(A,eltype(C),tuple(cdimsA...,odimsA...))
            tensorcopy!(Anew,vcat(clabels,olabelsA),A,labelsA)
        elseif conjA=='N'
            conjA='T' # it is more efficient to compute At*B
            Anew=similar(A,eltype(C),tuple(cdimsA...,odimsA...))
            tensorcopy!(Anew,vcat(clabels,olabelsA),A,labelsA)
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end

        # permute B
        if conjB=='C'
            Bnew=similar(B,eltype(C),tuple(odimsB...,cdimsB...))
            tensorcopy!(Bnew,vcat(olabelsB,clabels),B,labelsB)
        elseif conjB=='N'
            Bnew=similar(B,eltype(C),tuple(cdimsB...,odimsB...))
            tensorcopy!(Bnew,vcat(clabels,olabelsB),B,labelsB)
        else
            throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
        end
        
        # calculate C
        totalodimsA=prod(odimsA)
        totalodimsB=prod(odimsB)
        totalcdims=prod(cdimsA)
        
        Cnew=similar(C,tuple(odimsA...,odimsB...))
        if conjA=='T' && conjB=='N'
            At_mul_B!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalcdims,totalodimsB)))
        elseif conjA=='C' && conjB=='N'
            Ac_mul_B!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalcdims,totalodimsB)))
        elseif conjA=='T' && conjB=='C'
            At_mul_Bc!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalodimsB,totalcdims)))
        else
            Ac_mul_Bc!(reshape(Cnew,(totalodimsA,totalodimsB)),reshape(Anew,(totalcdims,totalodimsA)),reshape(Bnew,(totalodimsB,totalcdims)))
        end
        tensoradd!(beta,C,labelsC,alpha,Cnew,vcat(olabelsA,olabelsB))
        return C
    elseif method==:native
        stridesA=strides(A)
        stridesB=strides(B)
        stridesC=strides(C)
        ostridesA=stridesA[oindA]
        cstridesA=stridesA[cindA]
        ostridesB=stridesB[oindB]
        cstridesB=stridesB[cindB]
        ostridesCA=stridesC[oindCA]
        ostridesCB=stridesC[oindCB]

        unsafe_tensorcontract!(convert(T,beta),pointer(C),convert(T,alpha),pointer(A),conjA,pointer(B),conjB,ostridesCA,ostridesCB,ostridesA,cstridesA,ostridesB,cstridesB,odimsA,odimsB,cdimsA)
        return C
        
    else method==:buffered
        # to be written: allocate a fixed buffer, that can be used as temporary space for BLAS
    end
end

function tensorcontract{T1,T2}(A::StridedArray{T1},labelsA,B::StridedArray{T2},labelsB)
    dimsA=size(A)
    dimsB=size(A)
    labelsC=symdiff(labelsA,labelsB)
    if isempty(labelsC)
        return tensordot(A,labelsA,B,labelsB)
    else
        dimsC=Array(Int,length(labelsC))
        for (i,l)=enumerate(labelsC)
            ind=findfirst(labelsA,l)
            if ind>0
                dimsC[i]=dimsA[ind]
            else
                dimsC[i]=dimsB[findfirst(labelsB,l)]
            end
        end
        T=promote_type(T1,T2)
        C=similar(A,T,tuple(dimsC...))
        return tensorcontract!(zero(T),C,labelsC,one(T),A,labelsA,'N',B,labelsB,'N')
    end
end

let _tensorcontract_defined = Dict{(Int,Int,Int), Bool}()
    global unsafe_tensorcontract!
    function unsafe_tensorcontract!{T,TA,TB,N1,N2,N3}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},conjA::Char,B::Ptr{TB},conjB::Char,ostridesCA::NTuple{N1,Int},ostridesCB::NTuple{N2,Int},ostridesA::NTuple{N1,Int},cstridesA::NTuple{N3,Int},ostridesB::NTuple{N2,Int},cstridesB::NTuple{N3,Int},odimsA::NTuple{N1,Int},odimsB::NTuple{N2,Int},cdims::NTuple{N3,Int},obdimsA::NTuple{N1,Int},obdimsB::NTuple{N2,Int},cbdims::NTuple{N3,Int})
        def = get(_tensorcontract_defined,(N1,N2,N3),false)
        if !def
            ex = quote
            function _unsafe_tensorcontract!{T,TA,TB}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},conjA::Char,B::Ptr{TB},conjB::Char,ostridesCA::NTuple{$N1,Int},ostridesCB::NTuple{$N2,Int},ostridesA::NTuple{$N1,Int},cstridesA::NTuple{$N3,Int},ostridesB::NTuple{$N2,Int},cstridesB::NTuple{$N3,Int},odimsA::NTuple{$N1,Int},odimsB::NTuple{$N2,Int},cdims::NTuple{$N3,Int},obdimsA::NTuple{$N1,Int},obdimsB::NTuple{$N2,Int},cbdims::NTuple{$N3,Int})
                elsz=sizeof(T)
                elszA=sizeof(TA)
                elszB=sizeof(TB)
                
                conjA=='N' || conjA=='C' || throw(ArgumentError("invalid conjugation specification"))
                conjB=='N' || conjB=='C' || throw(ArgumentError("invalid conjugation specification"))
                
                # calculate dims as variables
                @nexprs $N1 d->(odimsA_{d} = odimsA[d])
                @nexprs $N2 d->(odimsB_{d} = odimsB[d])
                @nexprs $N3 d->(cdims_{d} = cdims[d])
                @nexprs $N1 d->(obdimsA_{d} = obdimsA[d])
                @nexprs $N2 d->(obdimsB_{d} = obdimsB[d])
                @nexprs $N2 d->(cbdims_{d} = cbdims[d])
                # calculate strides as variables
                @nexprs $N1 d->(ostridesA_{d} = ostridesA[d])
                @nexprs $N3 d->(cstridesA_{d} = cstridesA[d])
                @nexprs $N2 d->(ostridesB_{d} = ostridesB[d])
                @nexprs $N3 d->(cstridesB_{d} = cstridesB[d])
                @nexprs $N1 d->(ostridesCA_{d} = ostridesCA[d])
                @nexprs $N2 d->(ostridesCB_{d} = ostridesCB[d])
    
                @nexprs 1 d->(indA1_{$N1} = 1)
                @nexprs 1 d->(indC1_{$N1} = 1)
                @nloops($N1, outeri, d->1:obdimsA_{d}:odimsA_{d},
                    d->(indA1_{d-1}=indA1_{d};indC1_{d-1}=indC1_{d};ilim_{d}=min(outeri_{d}+obdimsA_{d}-1,odimsA_{d})), # PRE
                    d->(indA1_{d}+=obdimsA_{d}*ostridesA_{d};indC1_{d}+=obdimsA_{d}*ostridesCA_{d}), # POST
                    begin # BODY
                        @nexprs 1 e->(indC2_{$N2} = indC1_1)
                        @nexprs 1 e->(indB1_{$N2} = 1)
                        @nloops($N2, outerj, e->1:obdimsB_{e}:odimsB_{e},
                            e->(indC2_{e-1}=indC2_{e};indB1_{e-1}=indB1_{e};jlim_{e}=min(outerj_{e}+obdimsB_{e}-1,odimsB_{e})), # PRE
                            e->(indC2_{e}+=obdimsB_{e}*ostridesCB_{e};indB1_{e}+=obdimsB_{e}*ostridesB_{e}), # POST
                            begin # BODY
                                @nexprs 1 f->(indA2_{$N3} = indA1_1)
                                @nexprs 1 f->(indB2_{$N3} = indB1_1)
                                gamma=beta
                                @nloops($N3, outerk, f->1:cbdims_{f}:cdims_{f},
                                    f->(indA2_{f-1}=indA2_{f};indB2_{f-1}=indB2_{f};klim_{f}=min(outerk_{f}+cbdims_{f}-1,cdims_{f})), # PRE
                                    f->(indA2_{f}+=cbdims_{f}*cstridesA_{f};indB2_{f}+=cbdims_{f}*cstridesB_{f}), # POST
                                    begin # BODY
                                        @nexprs 1 g->(indA3_{$N1} = indA2_1)
                                        @nexprs 1 g->(indC3_{$N1} = indC2_1)
                                        @nloops($N1, inneri, a->outeri_{a}:ilim_{a},
                                            a->(indA3_{a-1} = indA3_{a};indC3_{a-1}=indC3_{a}), # PRE
                                            a->(indA3_{a} += ostridesA_{a};indC3_{a} += ostridesCA_{a}), # POST
                                            begin # BODY
                                                @nexprs 1 g->(indB3_{$N2} = indB2_1)
                                                @nexprs 1 g->(indC4_{$N2} = indC3_1)
                                                @nloops($N2, innerj, b->outerj_{b}:jlim_{b},
                                                    b->(indB3_{b-1} = indB3_{b};indC4_{b-1}=indC4_{b}), # PRE
                                                    b->(indB3_{b} += ostridesB_{b};indC4_{b} += ostridesCB_{b}), # POST
                                                    begin # BODY
                                                        @nexprs 1 f->(indA4_{$N3} = indA3_1)
                                                        @nexprs 1 f->(indB4_{$N3} = indB3_1)
                                                        localC=gamma*unsafe_load(C,indC4_1)
                                                        @nloops($N3, innerk, c->outerk_{c}:klim_{c},
                                                            c->(indA4_{c-1} = indA4_{c};indB4_{c-1}=indB4_{c}), # PRE
                                                            c->(indA4_{c} += cstridesA_{c};indB4_{c} += cstridesB_{c}), # POST
                                                            begin # BODY
                                                                localA=(conjA=='C' ? conj(unsafe_load(A,indA4_1)) : unsafe_load(A,indA4_1))
                                                                localB=(conjB=='C' ? conj(unsafe_load(B,indB4_1)) : unsafe_load(B,indB4_1))
                                                                localC+=alpha*localA*localB
                                                            end)
                                                        unsafe_store!(C,localC,indC4_1)
                                                    end)
                                            end)
                                        gamma=one(T)
                                    end)
                            end)
                    end)
                return C
            end
            end
            eval(current_module(),ex)
            _tensorcontract_defined[(N1,N2,N3)] = true
        end
        @time _unsafe_tensorcontract!(beta,C,alpha,A,conjA,B,conjB,ostridesCA,ostridesCB,ostridesA,cstridesA,ostridesB,cstridesB,odimsA,odimsB,cdims,obdimsA,obdimsB,cbdims)
    end
end
function unsafe_tensorcontract!{T,TA,TB,N1,N2,N3}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},conjA::Char,B::Ptr{TB},conjB::Char,ostridesCA::NTuple{N1,Int},ostridesCB::NTuple{N2,Int},ostridesA::NTuple{N1,Int},cstridesA::NTuple{N3,Int},ostridesB::NTuple{N2,Int},cstridesB::NTuple{N3,Int},odimsA::NTuple{N1,Int},odimsB::NTuple{N2,Int},cdims::NTuple{N3,Int})
    # Look for cache-friendly blocking strategy:
    elszC=sizeof(T)
    elszA=sizeof(TA)
    elszB=sizeof(TB)
    effectivecachesize=ifloor(cachesize/1.2) # 1.2 safety margin
    obstepA=ones(Int,N1)
    obstepB=ones(Int,N2)
    cbstep=ones(Int,N2)
    for i=1:N1
        obstepA[i]=max(1,div(cacheline,elszC*ostridesCA[i]),div(cacheline,elszA*ostridesA[i]))
    end
    for i=1:N2
        obstepB[i]=max(1,div(cacheline,elszC*ostridesCB[i]),div(cacheline,elszB*ostridesB[i]))
    end
    for i=1:N3
        cbstep[i]=max(1,div(cacheline,elszA*cstridesA[i]),div(cacheline,elszB*cstridesB[i]))
    end
    obdimsA=zeros(Int,N1)
    obdimsB=zeros(Int,N2)
    cbdims=zeros(Int,N3)
    cachesizeA=elszA*min(minimum(ostridesA),minimum(cstridesA))
    cachesizeB=elszB*min(minimum(ostridesB),minimum(cstridesB))
    cachesizeC=elszC*min(minimum(ostridesCA),minimum(ostridesCB))
    while true
        i=indmin(obdimsA)
        j=indmin(obdimsB)
        k=indmin(cbdims)
        if obdimsA[i]<=obdimsB[j] && obdimsA[i]<=cbdims[k]
            obdimsA[i]+=obstepA[i]
            if cachesizeC*prod(obdimsA)*prod(obdimsB)+cachesizeA*prod(obdimsA)*prod(cbdims)+cachesizeB*prod(obdimsB)*prod(cbdims)>effectivecachesize
                obdimsA[i]-=obstepA[i]
                break
            end
        elseif obdimsB[j]<=obdimsA[i] && obdimsB[j]<=cbdims[k]
            obdimsB[j]+=obstepB[j]
            if cachesizeC*prod(obdimsA)*prod(obdimsB)+cachesizeA*prod(obdimsA)*prod(cbdims)+cachesizeB*prod(obdimsB)*prod(cbdims)>effectivecachesize
                obdimsB[j]-=obstepB[j]
                break
            end
        else
            cbdims[k]+=cbstep[k]
            if cachesizeC*prod(obdimsA)*prod(obdimsB)+cachesizeA*prod(obdimsA)*prod(cbdims)+cachesizeB*prod(obdimsB)*prod(cbdims)>effectivecachesize
                cbdims[i]-=cbstep[k]
                break
            end
        end
    end
    unsafe_tensorcontract!(beta,C,alpha,A,conjA,B,conjB,ostridesCA,ostridesCB,ostridesA,cstridesA,ostridesB,cstridesB,odimsA,odimsB,cdims,tuple(obdimsA...),tuple(obdimsB...),tuple(cbdims...))
end

# const cachebuf = Array(Uint8, cachesize)
# let _tensorcontract_defined = Dict{(Int,Int,Int), Bool}()
#     global unsafe_tensorcontract!
#     function unsafe_tensorcontract!{T,TA,TB,N1,N2,N3}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},B::Ptr{TB},ostridesCA::NTuple{N1,Int},ostridesCB::NTuple{N2,Int},ostridesA::NTuple{N1,Int},cstridesA::NTuple{N3,Int},ostridesB::NTuple{N2,Int},cstridesB::NTuple{N3,Int},odimsA::NTuple{N1,Int},odimsB::NTuple{N2,Int},cdims::NTuple{N3,Int},obdimsA::NTuple{N1,Int},obdimsB::NTuple{N2,Int},cbdims::NTuple{N3,Int})
#         def = get(_tensorcontract_defined,(N1,N2,N3),false)
#         if !def
#             ex = quote
#             function _unsafe_tensorcontract!{T,TA,TB}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},B::Ptr{TB},ostridesCA::NTuple{$N1,Int},ostridesCB::NTuple{$N2,Int},ostridesA::NTuple{$N1,Int},cstridesA::NTuple{$N3,Int},ostridesB::NTuple{$N2,Int},cstridesB::NTuple{$N3,Int},odimsA::NTuple{$N1,Int},odimsB::NTuple{$N2,Int},cdims::NTuple{$N3,Int},obdimsA::NTuple{$N1,Int},obdimsB::NTuple{$N2,Int},cbdims::NTuple{$N3,Int})
#                 elsz=sizeof(T)
#                 elszA=sizeof(TA)
#                 elszB=sizeof(TB)
#                 
#                 # # set up cache tiles
#                 pCtile = convert(Ptr{T},pointer(cachebuf))
#                 pAtile = convert(Ptr{TA},pCtile+elsz*prod(obdimsA)*prod(obdimsB))
#                 pBtile = convert(Ptr{TB},pAtile+elszA*prod(obdimsA)*prod(cbdims))
#                     
#                 # calculate dims as variables
#                 @nexprs $N1 d->(odimsA_{d} = odimsA[d])
#                 @nexprs $N2 d->(odimsB_{d} = odimsB[d])
#                 @nexprs $N3 d->(cdims_{d} = cdims[d])
#                 @nexprs $N1 d->(obdimsA_{d} = obdimsA[d])
#                 @nexprs $N2 d->(obdimsB_{d} = obdimsB[d])
#                 @nexprs $N2 d->(cbdims_{d} = cbdims[d])
#                 # calculate strides as variables
#                 @nexprs $N1 d->(ostridesA_{d} = ostridesA[d])
#                 @nexprs $N3 d->(cstridesA_{d} = cstridesA[d])
#                 @nexprs $N2 d->(ostridesB_{d} = ostridesB[d])
#                 @nexprs $N3 d->(cstridesB_{d} = cstridesB[d])
#                 @nexprs $N1 d->(ostridesCA_{d} = ostridesCA[d])
#                 @nexprs $N2 d->(ostridesCB_{d} = ostridesCB[d])
#     
#                 @nexprs 1 d->(indA1_{$N1} = 1)
#                 @nexprs 1 d->(indC1_{$N1} = 1)
#                 @nloops($N1, outeri, d->1:obdimsA_{d}:odimsA_{d},
#                     d->(indA1_{d-1}=indA1_{d};indC1_{d-1}=indC1_{d};ilim_{d}=min(outeri_{d}+obdimsA_{d}-1,odimsA_{d})), # PRE
#                     d->(indA1_{d}+=obdimsA_{d}*ostridesA_{d};indC1_{d}+=obdimsA_{d}*ostridesCA_{d}), # POST
#                     begin # BODY
#                         tile_odimsA=@ntuple $N1 e->(ilim_{e}-outeri_{e}+1)
#                         tile_odimsAtot=prod(tile_odimsA)
#                         @nexprs 1 e->(indC2_{$N2} = indC1_1)
#                         @nexprs 1 e->(indB1_{$N2} = 1)
#                         @nloops($N2, outerj, e->1:obdimsB_{e}:odimsB_{e},
#                             e->(indC2_{e-1}=indC2_{e};indB1_{e-1}=indB1_{e};jlim_{e}=min(outerj_{e}+obdimsB_{e}-1,odimsB_{e})), # PRE
#                             e->(indC2_{e}+=obdimsB_{e}*ostridesCB_{e};indB1_{e}+=obdimsB_{e}*ostridesB_{e}), # POST
#                             begin # BODY
#                                 tile_odimsB=@ntuple $N2 f->(jlim_{f}-outerj_{f}+1)
#                                 tile_odimsBtot=prod(tile_odimsB)
#                                 Ctile=pointer_to_array(pCtile,tuple(tile_odimsA...,tile_odimsB...))
#                                 gamma=beta
#                                 @nexprs 1 f->(indA2_{$N3} = indA1_1)
#                                 @nexprs 1 f->(indB2_{$N3} = indB1_1)
#                                 @nloops($N3, outerk, f->1:cbdims_{f}:cdims_{f},
#                                     f->(indA2_{f-1}=indA2_{f};indB2_{f-1}=indB2_{f};klim_{f}=min(outerk_{f}+cbdims_{f}-1,cdims_{f})), # PRE
#                                     f->(indA2_{f}+=cbdims_{f}*cstridesA_{f};indB2_{f}+=cbdims_{f}*cstridesB_{f}), # POST
#                                     begin # BODY
#                                         tile_cdims=@ntuple $N3 g->(klim_{g}-outerk_{g}+1)
#                                         tile_cdimstot=prod(tile_cdims)
#                                         Atile=pointer_to_array(pAtile,tuple(tile_cdims...,tile_odimsA...))
#                                         Btile=pointer_to_array(pBtile,tuple(tile_cdims...,tile_odimsB...))
#                                         unsafe_tensorcopy!(pAtile,A+elszA*indA2_1,strides(Atile),tuple(cstridesA...,ostridesA...),tuple(tile_cdims...,tile_odimsA...),tuple(tile_cdims...,tile_odimsA...))
#                                         unsafe_tensorcopy!(pBtile,B+elszB*indB2_1,strides(Btile),tuple(cstridesB...,ostridesB...),tuple(tile_cdims...,tile_odimsB...),tuple(tile_cdims...,tile_odimsB...))
#                                         At_mul_B!(reshape(Ctile,(tile_odimsAtot,tile_odimsBtot)),reshape(Atile,(tile_cdimstot,tile_odimsAtot)),reshape(Btile,(tile_cdimstot,tile_odimsBtot)))
#                                         unsafe_tensoradd!(gamma,C,alpha,pCtile,tuple(ostridesCA...,ostridesCB...),strides(Ctile),tuple(tile_odimsA...,tile_odimsB...),tuple(tile_odimsA...,tile_odimsB...))
#                                     end)
#                             end)
#                     end)
#                 return C
#             end
#             end
#             eval(current_module(),ex)
#             _tensorcontract_defined[(N1,N2,N3)] = true
#         end
#         @time _unsafe_tensorcontract!(beta,C,alpha,A,B,ostridesCA,ostridesCB,ostridesA,cstridesA,ostridesB,cstridesB,odimsA,odimsB,cdims,obdimsA,obdimsB,cbdims)
#     end
# end

