using TensorOperations
using Test

TensorOperations.backend(:Strided)
TensorOperations.allocator(:Julia)
include("methods.jl")
include("tensor.jl")

TensorOperations.backend(:StridedBLAS)
TensorOperations.allocator(:Julia)
include("methods.jl")
include("tensor.jl")

TensorOperations.backend(:Strided)
TensorOperations.allocator(:Cache)
include("methods.jl")
include("tensor.jl")

TensorOperations.backend(:StridedBLAS)
TensorOperations.allocator(:Cache)
include("methods.jl")
include("tensor.jl")
