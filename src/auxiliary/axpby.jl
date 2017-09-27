# auxiliary/axpby.jl
#
# Support for dispatching tensor operations to optimized methods

const Zero = Type{Val{0}}
const One = Type{Val{1}}

# converts the scaling factor for tensor operation into a constant
# of appropriate type so that the optimal method is chosen by dispatch.
# Typically, there are optimized methods for 0 and 1 factors that
# remove the overhead of multiplication by 1 or addition of 0, so
# these constants are transformed into Val{0} and Val{1}, respectively
Base.@pure _opfactor(x) = x == 0 || x == 1 ? Val{x} : x
Base.@pure _opfactor(x::Union{Zero,One}) = x

const _zero = _opfactor(0)
const _one = _opfactor(1)

# Simple wrapper for the α * x + β * y operation
axpby(α::One,  x, β::One,  y) = x+y
axpby(α::Zero, x, β::One,  y) = y
axpby(α::One,  x, β::Zero, y) = x
axpby(α::Zero, x, β::Zero, y) = zero(y)

axpby(α::One,  x, β,       y) = x+β*y
axpby(α::Zero, x, β,       y) = β*y
axpby(α,       x, β::Zero, y) = α*x
axpby(α,       x, β::One,  y) = α*x+y

axpby(α,       x, β,       y) = α*x+β*y
