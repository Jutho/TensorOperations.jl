using TensorOperations
using Test

TensorOperations.backend(:CUDA)
TensorOperations.allocator(:Julia)

include("cutensor.jl")