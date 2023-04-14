using TensorOperations
using Test

# @testset "nocache" verbose = true begin
    TensorOperations.backend(:TBLIS)
    TensorOperations.allocator(:Julia)
    @testset "methods" include("methods.jl")
    # @testset "tensor"  include("tensor.jl")
# end

# @testset "cache" verbose = true begin
#     TensorOperations.backend(:TBLIS)
#     TensorOperations.allocator(:Cache)
#     include("methods.jl")
#     include("tensor.jl")
# end
