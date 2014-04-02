# LEVEL 2: two sets of indices

# In place methods
function tensortrace!{T1,T2}(alpha::Number,A::StridedArray{T1},labelsA,beta::Number,C::StridedArray{T2},labelsC)
    if length(labelsA)!=ndims(A) || length(labelsC)!=ndims(C)
        throw(ArgumentError("number of labels incompatible with number of tensor indices"))
    end
    
    po=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    pc1=indexin(clabels,labelsA)
    pc2=similar(pc1)
    for i=1:length(clabels)
        pc1[i]=findfirst(labelsA,clabels[i])
        pc2[i]=findnext(labelsA,clabels[i],pc1[i]+1)
    end
    isperm(vcat(po,pc1,pc2)) || throw(ArgumentError("invalid label specification"))
    
    dims=size(A)
    for i = 1:length(po)
        dims[po[i]] == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    for i = 1:length(pc1)
        dims[pc1[i]] == dims[pc2[i]] || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    
    stridesA=strides(A)
    ostridesA=stridesA[po]
    cstridesA1=stridesA[pc1]
    cstridesA2=stridesA[pc2]
    ostridesC=strides(C)
    
    odims=dims[po]
    cdims=dims[pc1]
    
    unsafe_tensortrace!(odims,cdims,convert(T2,alpha),pointer(A),ostridesA,cstridesA1,cstridesA2,convert(T2,beta),pointer(C),ostridesC)
    return C
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