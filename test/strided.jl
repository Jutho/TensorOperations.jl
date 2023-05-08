using TensorOperations
using Test

enable_blas()
include("methods.jl")
include("tensor.jl")

disable_blas()
include("methods.jl")
include("tensor.jl")
