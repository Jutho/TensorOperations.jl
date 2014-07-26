macro stridedloops(N, itersym, dimsym, args...)
    _stridedloops(N, itersym, dimsym, args...)
end
function _stridedloops(N::Int, itersym::Symbol, dimsym::Symbol, args...)
    mod(length(args),3)==1 || error("Wrong number of arguments")
    body = args[end]
    ex = Expr(:escape, body)
    for i=1:3:length(args)-1
        ex=Cartesian.sreplace!(ex,args[i],Cartesian.inlineanonymous(args[i], 1))
    end
    for dim = 1:N
        itervar = Cartesian.inlineanonymous(itersym, dim)
        dimvar = Cartesian.inlineanonymous(dimsym, dim)
        preargs = {}
        postargs = {}
        for i=1:3:length(args)-1
            indnew = Cartesian.inlineanonymous(args[i], dim)
            start = (dim < N ? Cartesian.inlineanonymous(args[i], dim+1) : args[i+1])
            step = Cartesian.inlineanonymous(args[i+2], dim)
            push!(preargs,:($(esc(indnew)) = $(esc(start))))
            push!(postargs,:($(esc(indnew)) += $(esc(step))))
        end
        preexpr=Expr(:block,preargs...)
        postexpr=Expr(:block,postargs...)
        ex = quote
            $preexpr
            for $(esc(itervar)) = 1:$(esc(dimvar))
                $ex
                $postexpr
            end
        end
    end
    if N == 0 # even a zero-dimensional array has one element and thus requires a single run
        preargs = {}
        for i=1:3:length(args)-1
            indnew = Cartesian.inlineanonymous(args[i], 1)
            start = args[i+1]
            push!(preargs,:($(esc(indnew)) = $(esc(start))))
        end
        preexpr=Expr(:block,preargs...)
        ex = quote
            $preexpr
            $ex
        end
    end
    ex
end

macro gentracekernel(N1,N2,order,alpha,A,beta,C,startA,startC,odims,cdims,ostridesA,cstridesA,ostridesC)
    _gentracekernel(N1,N2,order,alpha,A,beta,C,startA,startC,odims,cdims,ostridesA,cstridesA,ostridesC)
end
function _gentracekernel(N1::Int,N2::Int,order::Symbol,alpha::Symbol,A::Symbol,beta::Symbol,C::Symbol,
    startA::Symbol,startC::Symbol,odims::Symbol,cdims::Symbol,ostridesA::Symbol,cstridesA::Symbol,ostridesC::Symbol)
    ex=quote
        local indA1, indA2, indC
        local i,j
        if $(esc(order))==0
            local gamma
            gamma=$(esc(beta))
            @stridedloops($N1, i, $(esc(cdims)), indA1, $(esc(startA)), $(esc(cstridesA)), begin
                @stridedloops($N2, j, $(esc(odims)), indA2, indA1, $(esc(ostridesA)), indC, $(esc(startC)), $(esc(ostridesC)), begin
                    @inbounds $(esc(C))[indC]=gamma*$(esc(C))[indC]+$(esc(alpha))*$(esc(A))[indA2]
                end)
                gamma=one($(esc(beta)))
            end)
        else
            @stridedloops($N2, j, $(esc(odims)), indA1, $(esc(startA)), $(esc(ostridesA)), indC, $(esc(startC)), $(esc(ostridesC)), begin
                local localC
                @inbounds localC=$(esc(beta))*$(esc(C))[indC]
                @stridedloops($N1, i, $(esc(cdims)), indA2, indA1, $(esc(cstridesA)), @inbounds localC+=$(esc(alpha))*$(esc(A))[indA2])
                @inbounds $(esc(C))[indC]=localC
            end)
        end
    end
    ex
end

macro gencontractkernel(N1,N2,N3,order,alpha,A,conjA,B,conjB,beta,C,startA,startB,startC,odimsA,odimsB,cdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
    _gencontractkernel(N1,N2,N3,order,alpha,A,conjA,B,conjB,beta,C,startA,startB,startC,odimsA,odimsB,cdims,ostridesA,cstridesA,ostridesB,cstridesB,ostridesCA,ostridesCB)
end
function _gencontractkernel(N1::Int,N2::Int,N3::Int,order::Symbol,
    alpha::Symbol,A::Symbol,conjA::Symbol,B::Symbol,conjB::Symbol,beta::Symbol,C::Symbol,
    startA::Symbol,startB::Symbol,startC::Symbol,odimsA::Symbol,odimsB::Symbol,cdims::Symbol,
    ostridesA::Symbol,cstridesA::Symbol,ostridesB::Symbol,cstridesB::Symbol,ostridesCA::Symbol,ostridesCB::Symbol)
    ex=quote
        local indA1, indA2, indB1, indB2, indC1, indC2
        # we still have to implement other orders
#        if $(esc(order))==0 # i,j,k
            @stridedloops($N1, i, $(esc(odimsA)), indA1, $(esc(startA)), $(esc(ostridesA)), indC1, $(esc(startC)), $(esc(ostridesCA)), begin
                @stridedloops($N2, j, $(esc(odimsB)), indB1, $(esc(startB)), $(esc(ostridesB)), indC2, indC1, $(esc(ostridesCB)), begin
                    @inbounds localC=$(esc(beta))*$(esc(C))[indC2]
                    @stridedloops($N3, k, $(esc(cdims)), indA2, indA1, $(esc(cstridesA)), indB2, indB1, $(esc(cstridesB)), begin
                        @inbounds localA=($(esc(conjA))=='C' ? conj($(esc(A))[indA2]) : $(esc(A))[indA2])
                        @inbounds localB=($(esc(conjB))=='C' ? conj($(esc(B))[indB2]) : $(esc(B))[indB2])
                        localC+=$(esc(alpha))*localA*localB
                    end)
                    @inbounds $(esc(C))[indC2]=localC
                end)
            end)
#        end
    end
    ex
end
