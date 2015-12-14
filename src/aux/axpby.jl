# aux/axpby.jl
#
# Simple wrapper for the operation of computing α * x + β * y, and two
# special singleton types (immutables) to remove any overhead of multiplication
# by one or addition by zero.

immutable Zero <: Integer
end
immutable One <: Integer
end

const _zero = Zero()
const _one = One()

axpby(α::One,  x, β::One,  y) = x+y
axpby(α::Zero, x, β::One,  y) = y
axpby(α::One,  x, β::Zero, y) = x
axpby(α::Zero, x, β::Zero, y) = zero(y)

axpby(α::One,  x, β,       y) = x+β*y
axpby(α::Zero, x, β,       y) = β*y
axpby(α,       x, β::Zero, y) = α*x
axpby(α,       x, β::One,  y) = α*x+y

axpby(α,       x, β,       y) = α*x+β*y
