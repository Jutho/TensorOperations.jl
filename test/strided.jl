using TensorOperations
using Test

@testset "Strided noblas, nocache" verbose = true begin
    TensorOperations.backend(:Strided)
    TensorOperations.allocator(:Julia)
    include("methods.jl")
    include("tensor.jl")
end

@testset "Strided blas, nocache" verbose = true begin
    TensorOperations.backend(:StridedBLAS)
    TensorOperations.allocator(:Julia)
    include("methods.jl")
    include("tensor.jl")
end

@testset "Strided noblas, cache" verbose = true begin
    TensorOperations.backend(:Strided)
    TensorOperations.allocator(:Cache)
    include("methods.jl")
    include("tensor.jl")
end

@testset "Strided blas, cache" verbose = true begin
    TensorOperations.backend(:StridedBLAS)
    TensorOperations.allocator(:Cache)
    include("methods.jl")
    include("tensor.jl")
end
