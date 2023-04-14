using TensorOperations
using DynamicPolynomials
using Test

# Should probably make Polynomials compatible with the full VectorInterface
using VectorInterface: VectorInterface
function VectorInterface.scalartype(::Type{T}) where {T<:Union{<:PolyVar,<:Monomial,
                                                               <:Polynomial,<:Term}}
    return T
end

TensorOperations.backend(:Strided)
@polyvar a[1:2, 1:2] b[1:2, 1:2]
@tensor c[i, k] := a[i, j] * b[j, k]
@tensor d[i, k] := a[i, j] * b[j, k] + 2 * a[i, k]
@test c == a * b
@test d == a * b + 2 * a