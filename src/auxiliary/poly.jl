# auxiliary/poly.jl
#
# A lightweight polynomial implementation for abstract cost estimation
abstract AbstractPoly{T<:Number}

function Base.show{T<:Real}(io::IO,p::AbstractPoly{T})
    N = degree(p)
    first = true
    for i = N:-1:1
        if p[i]!=0
            if first
                p[i]==-1 && print(io, "-")
                abs(p[i])!=1 && print(io, "$(p[i])*")
            else
                print(io, p[i]<0 ? " - " : " + ")
                abs(p[i])==1 || print(io, "$(abs(p[i]))*")
            end
            print(io,"x")
            i>1 && print(io,"^$i")
            first = false
        end
    end
    if first
        print(io,p[0])
    elseif p[0]!=0
        print(io, p[0]<0 ? " - " : " + ")
        print(io, "$(abs(p[0]))")
    end
end
function Base.show{T<:Complex}(io::IO,p::AbstractPoly{T})
    N = degree(p)
    first = true
    for i = N:-1:1
        if p[i]!=0
            if first
                p[i]==1 || print(io, "($(p[i]))*")
            else
                print(io, " + ")
                p[i]==1 || print(io, "($(p[i]))*")
            end
            print(io,"x")
            i>1 && print(io,"^$i")
            first = false
        end
    end
    if first
        print(io,p[0])
    elseif p[0]!=0
        print(io, " + ")
        print(io, "($(p[0]))")
    end
end

immutable Power{T} <: AbstractPoly{T}
    coeff::T
    N::Int
end
degree(p::Power)=p.N
Base.getindex{T}(p::Power{T},i::Int)=(i==p.N ? p.coeff : zero(T))
Base.convert(::Type{Power},coeff::Number)=Power(coeff,0)

Base.(:-)(p::Power)=Power(-p.coeff,p.N)
Base.(:*)(p::Power,s::Number)=Power(p.coeff*s,p.N)
Base.(:*)(s::Number,p::Power)=*(p,s)
Base.(:*)(p1::Power,p2::Power)=Power(p1.coeff*p2.coeff,p1.N+p2.N)
Base.(:^)(p::Power,n::Int)=Power(p.coeff^n,n*p.N)

immutable Poly{T} <: AbstractPoly{T}
    coeffs::Vector{T}
end
degree(p::Poly)=length(p.coeffs)-1
Base.getindex{T}(p::Poly{T},i::Int)=(0<=i<=degree(p) ? p.coeffs[i+1] : zero(T))
Base.convert(::Type{Poly},coeff::Number)=Poly([coeff])

Base.(:+)(p::AbstractPoly, s::Number) = Poly([p[i]+(i==0 ? s : zero(s)) for i=0:degree(p)])
Base.(:+)(s::Number, p::AbstractPoly) = Poly([(i==0 ? s : zero(s))+p[i] for i=0:degree(p)])
Base.(:+)(p1::AbstractPoly,p2::AbstractPoly) = Poly([p1[i]+p2[i] for i=0:max(degree(p1),degree(p2))])

Base.(:-)(p::AbstractPoly, s::Number) = Poly([p[i]-(i==0 ? s : zero(s)) for i=0:degree(p)])
Base.(:-)(s::Number, p::AbstractPoly) = Poly([(i==0 ? s : zero(s))-p[i] for i=0:degree(p)])
Base.(:-)(p1::AbstractPoly,p2::AbstractPoly) = Poly([p1[i]-p2[i] for i=0:max(degree(p1),degree(p2))])

Base.(:*)(p1::Power,p2::Poly) = Poly([p1.coeff*p2[n-degree(p1)] for n=0:degree(p1)+degree(p2)])
Base.(:*)(p1::Poly,p2::Power) = Poly([p1[n-degree(p1)]*p2.coeff for n=0:degree(p1)+degree(p2)])
Base.(:*)(p::Poly,s::Number) = Poly(p.coeffs*s)
Base.(:*)(s::Number,p::Poly) = Poly(s*p.coeffs)
Base.(:*)(p1::Poly,p2::Poly)=begin
    N=degree(p1)+degree(p2)
    s=p1[0]*p2[0]
    coeffs=zeros(typeof(s),N+1)
    for i=0:degree(p1)
        for j=0:degree(p2)
            coeffs[i+j+1]+=p1[i]*p2[j]
        end
    end
    return Poly(coeffs)
end

function Base.(:(==))(p1::AbstractPoly,p2::AbstractPoly)
    for i=max(degree(p1),degree(p2)):-1:0
        p1[i]==p2[i] || return false
    end
    return true
end
function Base.(:<)(p1::AbstractPoly,p2::AbstractPoly)
    for i=max(degree(p1),degree(p2)):-1:0
        p1[i]<p2[i] && return true
        p1[i]>p2[i] && return false
    end
    return false
end
