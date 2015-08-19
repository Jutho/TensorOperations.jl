# axpby.jl
#
# Simple wrapper for the operation of computing alpha * x + beta * y, and two
# special singleton types (immutables) to remove any overhead of multiplication
# by one or addition by zero.

immutable Zero <: Integer
end
immutable One <: Integer
end

const _zero = Zero()
const _one = One()

axpby(a::One,  x, b::One,  y) = x+y
axpby(a::Zero, x, b::One,  y) = y
axpby(a::One,  x, b::Zero, y) = x
axpby(a::Zero, x, b::Zero, y) = zero(y)

axpby(a::One,  x, b,       y) = x+b*y
axpby(a::Zero, x, b,       y) = b*y
axpby(a,       x, b::Zero, y) = a*x
axpby(a,       x, b::One,  y) = a*x+y

axpby(a,       x, b,       y) = a*x+b*y
