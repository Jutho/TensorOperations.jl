using Test
using TensorOperations
using Random
using LinearAlgebra

include("gradients.jl"); println("============== DONE GRAD TESTS ==============")

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
