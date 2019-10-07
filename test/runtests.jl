using Test
using TensorOperations
using Random
using LinearAlgebra

Random.seed!(1234567)

TensorOperations.enable_blas()
TensorOperations.enable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperations.disable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperations.disable_blas()
include("methods.jl")
include("tensor.jl")

include("tensoropt.jl")
include("auxiliary.jl")
