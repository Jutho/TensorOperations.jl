using TensorOperations
using DynamicPolynomials
using Test

using VectorInterface: VectorInterface
const PolyTypes = Union{PolyVar,Monomial,Polynomial,Term}

VectorInterface.scalartype(::Type{T}) where {T<:PolyTypes} = T
VectorInterface.add!!(w::PolyTypes, v::PolyTypes, α::Number, β::Number) = w * β + v * α
VectorInterface.scale!!(v::PolyTypes, α::Number) = w * α

TensorOperations.backend(:Strided)
@polyvar a[1:2, 1:2] b[1:2, 1:2]
@tensor c[i, k] := a[i, j] * b[j, k]
@tensor d[i, k] := a[i, j] * b[j, k] + 2 * a[i, k]
@test c == a * b
@test d == a * b + 2 * a
