# lightweight poly for abstract cost estimation
abstract type AbstractPoly{D, T <: Number} end
Base.one(x::AbstractPoly) = one(typeof(x))
Base.zero(x::AbstractPoly) = zero(typeof(x))

function Base.show(io::IO, p::AbstractPoly{D, T}) where {D, T <: Real}
    N = degree(p)
    for i in N:-1:0
        if i > 0
            print(io, "$(abs(p[i]))*")
            print(io, "$D")
            i > 1 && print(io, "^$i")
            print(io, p[i - 1] < 0 ? " - " : " + ")
        else
            print(io, "$(abs(p[i]))")
        end
    end
    return
end
function Base.show(io::IO, p::AbstractPoly{D, T}) where {D, T <: Complex}
    N = degree(p)
    for i in N:-1:0
        if i > 0
            print(io, "($(p[i]))*")
            print(io, "$D")
            i > 1 && print(io, "^$i")
            print(io, " + ")
        else
            print(io, "($(p[i]))")
        end
    end
    return
end
struct Power{D, T} <: AbstractPoly{D, T}
    coeff::T
    N::Int
end
degree(p::Power) = p.N
Base.getindex(p::Power{D, T}, i::Int) where {D, T} = (i == p.N ? p.coeff : zero(p.coeff))
Power{D}(coeff::T, N::Int = 0) where {D, T} = Power{D, T}(coeff, N)

Base.one(::Type{Power{D, T}}) where {D, T} = Power{D, T}(one(T), 0)
Base.zero(::Type{Power{D, T}}) where {D, T} = Power{D, T}(zero(T), 0)

Base.convert(::Type{Power{D}}, coeff::Number) where {D} = Power{D}(coeff, 0)
Base.convert(::Type{Power{D, T}}, coeff::Number) where {D, T} = Power{D, T}(coeff, 0)
Base.convert(::Type{Power{D, T}}, p::Power{D}) where {D, T} = Power{D, T}(p.coeff, p.N)

function Base.show(io::IO, p::Power{D, T}) where {D, T}
    if p.coeff == 1
    elseif p.coeff == -1
        print(io, "-")
    elseif isa(p.coeff, Complex)
        print(io, "($(p.coeff))")
    else
        print(io, "$(p.coeff)")
    end
    p.coeff == 1 || p.coeff == -1 || p.N == 0 || print(io, "*")
    p.N == 0 && (p.coeff == 1 || p.coeff == -1) && print(io, "1")
    p.N > 0 && print(io, "$D")
    return p.N > 1 && print(io, "^$(p.N)")
end

function Base.:*(p1::Power{D}, p2::Power{D}) where {D}
    return Power{D}(p1.coeff * p2.coeff, degree(p1) + degree(p2))
end
Base.:*(p::Power{D}, s::Number) where {D} = Power{D}(p.coeff * s, degree(p))
Base.:*(s::Number, p::Power) = *(p, s)
Base.:/(p::Power{D}, s::Number) where {D} = Power{D}(p.coeff / s, degree(p))
Base.:\(s::Number, p::Power) = /(p, s)
Base.:^(p::Power{D}, n::Int) where {D} = Power{D}(p.coeff^n, n * degree(p))

struct Poly{D, T} <: AbstractPoly{D, T}
    coeffs::Vector{T}
    function Poly{D, T}(coeffs::Vector{T}) where {D, T}
        if length(coeffs) == 0 || coeffs[end] == 0
            i = findlast(!iszero, coeffs)
            return i === nothing ? new{D, T}(T[0]) : new{D, T}(coeffs[1:i])
        else
            return new{D, T}(coeffs)
        end
    end
end
degree(p::Poly) = max(0, length(p.coeffs) - 1)
function Base.getindex(p::Poly{D, T}, i::Int) where {D, T}
    return (0 <= i <= degree(p) ? p.coeffs[i + 1] : zero(p[0]))
end
Poly{D}(coeffs::Vector{T}) where {D, T} = Poly{D, T}(coeffs)
Poly{D}(c0::T) where {D, T} = Poly{D, T}([c0])
Poly{D}(p::Power{D, T}) where {D, T} = Poly{D, T}(vcat(zeros(T, p.N), p.coeff))
Poly{D, T}(c0::Number) where {D, T} = Poly{D, T}([T(c0)])
Poly{D, T1}(p::Power{D, T2}) where {D, T1, T2} = Poly{D, T1}(vcat(zeros(T1, p.N), T1(p.coeff)))

Base.one(::Type{Poly{D, T}}) where {D, T} = Poly{D, T}([one(T)])
Base.zero(::Type{Poly{D, T}}) where {D, T} = Poly{D, T}([zero(T)])

Base.convert(::Type{Poly{D}}, x::Number) where {D} = Poly{D}([x])
Base.convert(::Type{Poly{D, T}}, x::Number) where {D, T} = Poly{D, T}(T[x])
function Base.convert(::Type{Poly{D}}, p::Power{D}) where {D}
    return Poly{D}(vcat(fill(zero(p.coeff), p.N), p.coeff))
end
function Base.convert(::Type{Poly{D, T}}, p::Power{D}) where {D, T}
    return Poly{D, T}(vcat(fill(zero(T), p.N), convert(T, p.coeff)))
end
function Base.convert(::Type{Poly{D, T}}, p::Poly{D}) where {D, T}
    return Poly{D, T}(convert(Vector{T}, p.coeffs))
end

function Base.:+(p::AbstractPoly{D}, s::Number) where {D}
    return Poly{D}([p[i] + ifelse(i == 0, s, zero(s)) for i in 0:degree(p)])
end
Base.:+(s::Number, p::AbstractPoly) = +(p, s)
function Base.:+(p1::AbstractPoly{D}, p2::AbstractPoly{D}) where {D}
    return Poly{D}([p1[i] + p2[i] for i in 0:max(degree(p1), degree(p2))])
end

Base.:-(p::Poly{D}) where {D} = Poly{D}(-p.coeffs)
function Base.:-(p::AbstractPoly{D}, s::Number) where {D}
    return Poly{D}([p[i] - ifelse(i == 0, s, zero(s)) for i in 0:degree(p)])
end
function Base.:-(s::Number, p::AbstractPoly{D}) where {D}
    return Poly{D}([-p[i] + ifelse(i == 0, s, zero(s)) for i in 0:degree(p)])
end
function Base.:-(p1::AbstractPoly{D}, p2::AbstractPoly{D}) where {D}
    return Poly{D}([p1[i] - p2[i] for i in 0:max(degree(p1), degree(p2))])
end

function Base.:*(p1::Power{D}, p2::Poly{D}) where {D}
    return Poly{D}([p1.coeff * p2[n - degree(p1)] for n in 0:(degree(p1) + degree(p2))])
end
Base.:*(p1::Poly{D}, p2::Power{D}) where {D} = *(p2, p1)
Base.:*(p::Poly{D}, s::Number) where {D} = Poly{D}(s * p.coeffs)
Base.:*(s::Number, p::Poly) = *(p, s)
Base.:/(p::Poly{D}, s::Number) where {D} = Poly{D}(p.coeffs / s)
Base.:\(s::Number, p::Poly) = /(p, s)
function Base.:*(p1::Poly{D}, p2::Poly{D}) where {D}
    N = degree(p1) + degree(p2)
    s = p1[0] * p2[0]
    coeffs = zeros(typeof(s), N + 1)
    for i in 0:degree(p1)
        for j in 0:degree(p2)
            coeffs[i + j + 1] += p1[i] * p2[j]
        end
    end
    return Poly{D}(coeffs)
end

function Base.promote_rule(
        ::Type{Power{D, T1}}, ::Type{Power{D, T2}}
    ) where {D, T1 <: Number, T2 <: Number}
    return Power{D, promote_type(T1, T2)}
end
function Base.promote_rule(::Type{Power{D, T1}}, ::Type{T2}) where {D, T1 <: Number, T2 <: Number}
    return Power{D, promote_type(T1, T2)}
end
function Base.promote_rule(
        ::Type{Poly{D, T1}}, ::Type{Poly{D, T2}}
    ) where {D, T1 <: Number, T2 <: Number}
    return Poly{D, promote_type(T1, T2)}
end
function Base.promote_rule(
        ::Type{Poly{D, T1}}, ::Type{Power{D, T2}}
    ) where {D, T1 <: Number, T2 <: Number}
    return Poly{D, promote_type(T1, T2)}
end
function Base.promote_rule(::Type{Poly{D, T1}}, ::Type{T2}) where {D, T1 <: Number, T2 <: Number}
    return Poly{D, promote_type(T1, T2)}
end

function Base.:(==)(p1::AbstractPoly{D}, p2::AbstractPoly{D}) where {D}
    for i in max(degree(p1), degree(p2)):-1:0
        p1[i] == p2[i] || return false
    end
    return true
end
Base.:(==)(p1::AbstractPoly, p2::Number) = degree(p1) == 0 && p1[0] == p2
Base.:(==)(p1::Number, p2::AbstractPoly) = degree(p2) == 0 && p2[0] == p1
function Base.:<(p1::AbstractPoly{D}, p2::AbstractPoly{D}) where {D}
    for i in max(degree(p1), degree(p2)):-1:0
        p1[i] < p2[i] && return true
        p1[i] > p2[i] && return false
    end
    return false
end
Base.isless(p1::AbstractPoly{D}, p2::AbstractPoly{D}) where {D} = p1 < p2
