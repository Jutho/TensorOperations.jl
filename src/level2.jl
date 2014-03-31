# LEVEL 2: one set of open indices and one set of contracted indices

using Base.Cartesian


function tensorproject!{T,TA,TB}(beta::Number,C::StridedArray{T},labelsC,alpha::Number,A::StridedArray{TA},labelsA,B::StridedArray{TB},labelsB)
    if length(labelsA)!=ndims(A) || length(labelsB)!=ndims(B) || length(labelsC)!=ndims(C)
        throw(ArgumentError("number of labels incompatible with number of tensor indices"))
    end
    
    pc=indexin(labelsB,labelsA)
    po=indexin(labelsC,labelsA)
    
    isperm(vcat(po,pc)) || throw(ArgumentError("invalid label specification"))
    (isbits(T) && isbits(TA) && isbits(TB)) || error("only arrays of bitstypes are supported")
    
    dims=size(A)
    for i = 1:length(po)
        dims[po[i]] == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    for i = 1:length(pc)
        dims[pc[i]] == size(B,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    
    stridesA=strides(A)
    ostridesA=stridesA[po]
    cstridesA=stridesA[pc]
    cstridesB=strides(B)
    ostridesC=strides(C)
    
    odims=dims[po]
    cdims=dims[pc]
      
    unsafe_tensorproject!(convert(T,beta),pointer(C),convert(T,alpha),pointer(A),pointer(B),ostridesC,ostridesA,cstridesA,cstridesB,odims,cdims)
    return C
end

# function tensortrace!{T}(beta::Number,C::StridedArray{T},labelsC,alpha::Number,A::StridedArray{T},labelsA)
#     if length(labelsA)!=ndims(A) || length(labelsC)!=ndims(C)
#         throw(ArgumentError("number of labels incompatible with number of tensor indices"))
#     end
#     
#     po=indexin(labelsC,labelsA)
#     clabels=unique(setdiff(labelsA,labelsC))
#     pc1=indexin(clabels,labelsA)
#     pc2=similar(pc1)
#     for i=1:length(pc1)
#         pc1[i]=findfirst(labelsA,clabels[i])
#         pc2[i]=findnext(labelsA,clabels[i],pc1[i]+1)
#     end
#     isperm(vcat(po,pc1,pc2)) || throw(ArgumentError("invalid label specification"))
#     
#     dims=size(A)
#     for i = 1:length(po)
#         dims[po[i]] == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
#     end
#     for i = 1:length(pc1)
#         dims[pc1[i]] == dims[pc2[i]] || throw(DimensionMismatch("tensor sizes incompatible"))
#     end
#     
#     stridesA=strides(A)
#     ostridesA=stridesA[po]
#     cstridesA1=stridesA[pc1]
#     cstridesA2=stridesA[pc2]
#     ostridesC=strides(C)
#     
#     odims=dims[po]
#     cdims=dims[pc1]
#     
#     unsafe_tensortrace!(convert(T,beta),pointer(C),convert(T,alpha),pointer(A),ostridesC,ostridesA,cstridesA1,cstridesA2,odims,cdims)
#     return C
# end


# TENSORPROJECT
const cachesize = 1<<15
const cacheline = 64

let _tensorproject_defined = Dict{(Int,Int), Bool}()
    global unsafe_tensorproject!
    function unsafe_tensorproject!{T,TA,TB,N1,N2}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},B::Ptr{TB},ostridesC::NTuple{N1,Int},ostridesA::NTuple{N1,Int},cstridesA::NTuple{N2,Int},cstridesB::NTuple{N2,Int},odims::NTuple{N1,Int},cdims::NTuple{N2,Int},oblockdims::NTuple{N1,Int},cblockdims::NTuple{N2,Int})
        def = get(_tensorproject_defined,(N1,N2),false)
        if !def
            ex = quote
            function _unsafe_tensorproject!{T,TA,TB}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},B::Ptr{TB},ostridesC::NTuple{$N1,Int},ostridesA::NTuple{$N1,Int},cstridesA::NTuple{$N2,Int},cstridesB::NTuple{$N2,Int},odims::NTuple{$N1,Int},cdims::NTuple{$N2,Int},oblockdims::NTuple{$N1,Int},cblockdims::NTuple{$N2,Int})
                # calculate dims as variables
                @nexprs $N1 d->(odims_{d} = odims[d])
                @nexprs $N2 d->(cdims_{d} = cdims[d])
                @nexprs $N1 d->(obdims_{d} = oblockdims[d])
                @nexprs $N2 d->(cbdims_{d} = cblockdims[d])
                # calculate strides as variables
                @nexprs $N1 d->(ostridesA_{d} = ostridesA[d])
                @nexprs $N2 d->(cstridesA_{d} = cstridesA[d])
                @nexprs $N2 d->(cstridesB_{d} = cstridesB[d])
                @nexprs $N1 d->(ostridesC_{d} = ostridesC[d])
    
                @nexprs 1 d->(indA_{$N1} = 1)
                @nexprs 1 d->(indC_{$N1} = 1)
                @nloops($N1, outeri, d->1:obdims_{d}:odims_{d},
                    d->(indA_{d-1} = indA_{d};indC_{d-1}=indC_{d}), # PRE
                    d->(indA_{d} += obdims_{d}*ostridesA_{d};indC_{d} += obdims_{d}*ostridesC_{d}), # POST
                    begin # BODY
                        @nexprs 1 e->(indA2_{$N2} = indA_{1})
                        @nexprs 1 e->(indB_{$N2} = 1)
                        @nloops($N2, outerj, e->1:cbdims_{e}:cdims_{e},
                            e->(indA2_{e-1} = indA2_{e};indB_{e-1}=indB_{e}), # PRE
                            e->(indA2_{e} += cbdims_{e}*cstridesA_{e};indB_{e} += cbdims_{e}*cstridesB_{e}), # POST
                            begin # BODY
                                @nexprs 1 f->(indA3_{$N1} = indA2_{1})
                                @nexprs 1 f->(indC2_{$N1} = indC_{1})
                                @nloops($N1, inneri, f->outeri_{f}:min(outeri_{f}+obdims_{f}-1,odims_{f}),
                                    f->(indA3_{f-1} = indA3_{f};indC2_{f-1}=indC2_{f}), # PRE
                                    f->(indA3_{f} += ostridesA_{f};indC2_{f} += ostridesC_{f}), # POST
                                    begin # BODY
                                        localC::T=beta*unsafe_load(C,indC2_1)
                                        @nexprs 1 g->(indA4_{$N2} = indA3_{1})
                                        @nexprs 1 g->(indB2_{$N2} = indB_{1})
                                        @nloops($N2, innerj, g->outerj_{g}:min(outerj_{g}+cbdims_{g}-1,cdims_{g}),
                                            g->(indA4_{g-1} = indA4_{g};indB2_{g-1}=indB2_{g}), # PRE
                                            g->(indA4_{g} += cstridesA_{g};indB2_{g} += cstridesB_{g}), # POST
                                            localC+=alpha*unsafe_load(A,indA4_1)*unsafe_load(B,indB2_1)) #BODY
                                        unsafe_store!(C,localC,indC2_1)
                                    end)
                            end)
                    end)
                return C
            end
            end
            eval(current_module(),ex)
            _tensorproject_defined[(N1,N2)] = true
        end
        _unsafe_tensorproject!(beta,C,alpha,A,B,ostridesC,ostridesA,cstridesA,cstridesB,odims,cdims,oblockdims,cblockdims)
    end
function unsafe_tensorproject!{T,TA,TB,N1,N2}(beta::T,C::Ptr{T},alpha::T,A::Ptr{TA},B::Ptr{TB},ostridesC::NTuple{N1,Int},ostridesA::NTuple{N1,Int},cstridesA::NTuple{N2,Int},cstridesB::NTuple{N2,Int},odims::NTuple{N1,Int},cdims::NTuple{N2,Int})
    # Look for cache-friendly blocking strategy:
    elszC=sizeof(T)
    elszA=sizeof(TA)
    elszB=sizeof(TB)
    # effectivecachesize=ifloor(cachesize/1.4) # 1.4 safety margin
    # oblockstep=ones(Int,N1)
    # cblockstep=ones(Int,N2)
    # for i=1:N1
    #     oblockstep[i]=max(1,div(cacheline,elszC*ostridesC[i]),div(cacheline,elszA*ostridesA[i]))
    # end
    # for i=1:N2
    #     cblockstep[i]=max(1,div(cacheline,elszA*cstridesA[i]),div(cacheline,elszB*cstridesB[i]))
    # end
    # oblockdims=zeros(Int,N1)
    # cblockdims=zeros(Int,N2)
    # i=1
    # j=1
    # while true
    #     oblockdims[i]+=oblockstep[i]
    #     if elszA*prod(oblockdims)*prod(cblockdims)+elszC*prod(oblockdims)+elszB*prod(cblockdims)>effectivecachesize
    #         oblockdims[i]-=oblockstep[i]
    #         break
    #     end
    #     i+=1
    #     if i>N1; i=1; end
    #     cblockdims[j]+=cblockstep[j]
    #     if elszA*prod(oblockdims)*prod(cblockdims)+elszC*prod(oblockdims)+elszB*prod(cblockdims)>effectivecachesize
    #         cblockdims[j]-=cblockstep[j]
    #         break
    #     end
    #     j+=1
    #     if j>N2; j=1; end
    # end
    # unsafe_tensorproject!(beta,C,alpha,A,B,ostridesC,ostridesA,cstridesA,cstridesB,odims,cdims,tuple(oblockdims...),tuple(cblockdims...))        
    unsafe_tensorproject!(beta,C,alpha,A,B,ostridesC,ostridesA,cstridesA,cstridesB,odims,cdims,ntuple(n->1,N1),ntuple(n->1,N2))        
end
end


# 
# let _tensortrace_defined = Dict{(Int,Int), Bool}()
#     global unsafe_tensortrace!
#     function unsafe_tensortrace!{T,N1,N2}(beta::T,C::Ptr{T},alpha::T,A::Ptr{T},ostridesC::NTuple{N1,Int},ostridesA::NTuple{N1,Int},cstridesA1::NTuple{N2,Int},cstridesA2::NTuple{N2,Int},odims::NTuple{N1,Int},cdims::NTuple{N2,Int})
#         def = get(_tensortrace_defined,(N1,N2),false)
#         if !def
#             ex = quote
#             function _unsafe_tensortrace!{T}(beta::T,C::Ptr{T},alpha::T,A::Ptr{T},ostridesC::NTuple{$N1,Int},ostridesA::NTuple{$N1,Int},cstridesA1::NTuple{$N2,Int},cstridesA2::NTuple{$N2,Int},odims::NTuple{$N1,Int},cdims::NTuple{$N2,Int})
#                 elsz = isbits(T) ? sizeof(T) : sizeof(Ptr)
#                 cachelength = ifloor(cachesize/(elsz*1.4)) # 1.4 safety margin
#                 
#                 # calculates all the strides as variables
#                 ostridesA_1 = 0
#                 cstridesA1_1 = 0
#                 cstridesA2_1 = 0
#                 ostridesC_1 = 0
#                 @nexprs $N1 d->(ostridesA_{d+1} = ostridesA[d])
#                 @nexprs $N2 d->(cstridesA1_{d+1} = cstridesA1[d])
#                 @nexprs $N2 d->(cstridesA2_{d+1} = cstridesA2[d])
#                 @nexprs $N1 d->(ostridesC_{d+1} = ostridesC[d])
#                 
#                 @nexprs $N1 d->(odims_{d} = odims[d])
#                 @nexprs $N2 d->(cdims_{d} = cdims[d])
# 
#                 if prod(odims)*(prod(cdims)*prod(cdims)+1) <= cachelength # odims*cdims^2 (for A) + odims (for C)
#                     # creates offset, because indexing starts at 1
#                     offsetA = 1 - sum(ostridesA) - sum(cstridesA1) - sum(cstridesA2)
#                     offsetC = 1 - sum(ostridesC)
#                     
#                     # set counts_{N1+1}
#                     @nexprs 1 d->(ocountsA_{$N1+1} = ostridesA_{$N1+1})
#                     @nexprs 1 d->(ocountsC_{$N1+1} = ostridesC_{$N1+1})
#                     
#                     @nloops($N1, i, d->1:odims_{d},
#                         d->(ocountsA_d = ostridesA_d;ocountsC_d=ostridesC_d), # PRE
#                         d->(ocountsA_{d+1} += ostridesA_{d+1};ocountsC_{d+1} += ostridesC_{d+1}), # POST
#                         begin # BODY
#                             indC = sum(@ntuple $N1 n->ocountsC_{n+1})+offsetC
#                             indA = sum(@ntuple $N1 n->ocountsA_{n+1})+offsetA
#                             # scale C
#                             unsafe_store!(C,beta*unsafe_load(C,indC),indC)
#                             # set counts_{N2+1}
#                             @nexprs 1 n->(ccountsA1_{$N2+1} = cstridesA1_{$N2+1})
#                             @nexprs 1 n->(ccountsA2_{$N2+1} = cstridesA2_{$N2+1})
#                             @nloops($N2, k, n->1:cdims_{n},
#                                 n->(ccountsA1_n = cstridesA1_n;ccountsA2_n=cstridesA2_n), # PRE
#                                 n->(ccountsA1_{n+1} += cstridesA1_{n+1};ccountsA2_{n+1} += cstridesA2_{n+1}), # POST
#                                 begin # BODY
#                                     indA1 = indA+sum(@ntuple $N2 m->ccountsA1_{m+1})+sum(@ntuple $N2 m->ccountsA2_{m+1})
#                                     unsafe_store!(C,unsafe_load(C,indC)+alpha*unsafe_load(A,indA1),indC)
#                                 end)
#                         end)
#                 else
#                     dimmax=1
#                     tobesplit=true
#                     @nexprs $N1 d->(if odims_{d}>dimmax; dimmax=odims_{d}; end)
#                     @nexprs $N2 d->(if cdims_{d}>dimmax; dimmax=cdims_{d}; end)
#                     newdim1=dimmax>>1
#                     newdim2=dimmax-newdim1
#                     
#                     if tobesplit && odims_1==dimmax
#                         odims_1=newdim1
#                         _unsafe_tensortrace!(beta,C,alpha,A,ostridesC,ostridesA,cstridesA!,cstridesA2,(@ntuple $N1 n->odims_{n}),cdims)
#                         odims_1=newdim2
#                         _unsafe_tensortrace!(beta,C+elsz*newdim1*ostridesC_2,alpha,A+elsz*newdim1*ostridesA_2,ostridesC,ostridesA,cstridesA1,cstridesA2,(@ntuple $N1 n->odims_{n}),cdims)
#                         tobesplit=false
#                     end
#                     @nif $N1 d->(tobesplit && odims_{d+1}==dimmax) d->begin
#                         odims_{d+1}=newdim1
#                         _unsafe_tensortrace!(beta,C,alpha,A,ostridesC,ostridesA,cstridesA1,cstridesA2,(@ntuple $N1 n->odims_{n}),cdims)
#                         odims_{d+1}=newdim2
#                         _unsafe_tensortrace!(beta,C+elsz*newdim1*ostridesC_{d+2},alpha,A+elsz*newdim1*ostridesA_{d+2},ostridesC,ostridesA,cstridesA1,cstridesA2,(@ntuple $N1 n->odims_{n}),cdims)
#                         tobesplit=false
#                     end d->()
#                     if tobesplit && cdims_1==dimmax
#                         cdims_1=newdim1
#                         _unsafe_tensortrace!(beta,C,alpha,A,ostridesC,ostridesA,cstridesA1,cstridesA2,odims,(@ntuple $N2 n->cdims_{n}))
#                         cdims_1=newdim2
#                         _unsafe_tensortrace!(one(T),C,alpha,A+elsz*newdim1*cstridesA1_2+elsz*newdim1*cstridesA2_2,ostridesC,ostridesA,cstridesA1,cstridesA2,odims,(@ntuple $N2 n->cdims_{n}))
#                         tobesplit=false
#                     end
#                     @nif $N2 d->(tobesplit && cdims_{d+1}==dimmax) d->begin
#                         cdims_{d+1}=newdim1
#                         _unsafe_tensortrace!(beta,C,alpha,A,ostridesC,ostridesA,cstridesA,cstridesB,odims,(@ntuple $N2 n->cdims_{n}))
#                         cdims_{d+1}=newdim2
#                         _unsafe_tensortrace!(one(T),C,alpha,A+elsz*newdim1*cstridesA1_{d+2}+elsz*newdim1*cstridesA2_{d+2},ostridesC,ostridesA,cstridesA1,cstridesA2,odims,(@ntuple $N2 n->cdims_{n}))
#                         tobesplit=false
#                     end d->()
#                 end
#                 return C
#             end
#         end
#         eval(current_module(),ex)
#         _tensortrace_defined[(N1,N2)] = true
#     end
#     _unsafe_tensortrace!(beta,C,alpha,A,ostridesC,ostridesA,cstridesA1,cstridesA2,odims,cdims)
# end
# end