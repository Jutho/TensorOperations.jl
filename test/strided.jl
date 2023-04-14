using TensorOperations
using Test

@testset "noblas, nocache" verbose=true begin
    TensorOperations.backend(:Strided)
    TensorOperations.allocator(:Julia)
    include("methods.jl")
    include("tensor.jl")
end

@testset "blas, nocache" verbose=true begin
    TensorOperations.backend(:StridedBLAS)
    TensorOperations.allocator(:Julia)
    include("methods.jl")
    include("tensor.jl")
end

@testset "noblas, cache" verbose=true begin
    TensorOperations.backend(:Strided)
    TensorOperations.allocator(:Cache)
    include("methods.jl")
    include("tensor.jl")
end

@testset "blas, cache" verbose=true begin
    TensorOperations.backend(:StridedBLAS)
    TensorOperations.allocator(:Cache)
    include("methods.jl")
    include("tensor.jl")
end
