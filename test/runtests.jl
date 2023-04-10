using Test
using TensorOperations, TensorOperationsCore
using Random
using LinearAlgebra
using CUDA, cuTENSOR

Random.seed!(1234567)

operationbackend!(StridedBackend(false))

include("methods.jl")
include("tensor.jl")

# TensorOperations.enable_blas()
# TensorOperations.enable_cache()
# include("methods.jl")



# include("tensor.jl")
# allocationbackend!(TensorCache())
# include("tensor.jl")

# TensorOperations.disable_cache()
# include("methods.jl")
# include("tensor.jl")
# TensorOperations.disable_blas()
# include("methods.jl")
# include("tensor.jl")
# TensorOperations.enable_blas()
# TensorOperations.enable_cache()

# if CUDA.functional()
#     include("cutensor.jl")
# end

# include("tensoropt.jl")
# include("auxiliary.jl")
