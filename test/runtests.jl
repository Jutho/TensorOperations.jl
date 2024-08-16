using TensorOperations
using LinearAlgebra
using Test
using Logging
using Random
Random.seed!(1234567)

using TensorOperations: IndexError
using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
using TensorOperations: DefaultAllocator, ManualAllocator

@testset "tensoropt" verbose = true begin
    include("tensoropt.jl")
end
@testset "auxiliary" verbose = true begin
    include("auxiliary.jl")
end
@testset "macro keywords" verbose = true begin
    include("macro_kwargs.jl")
end
@testset "method syntax" verbose = true begin
    include("methods.jl")
end
@testset "macro with index notation" verbose = true begin
    include("tensor.jl")
end
@testset "ad" verbose = false begin
    include("ad.jl")
end

# note: cuTENSOR should not be loaded before this point
# as there is a test which requires it to be loaded after
@testset "cuTENSOR extension" verbose = true begin
    include("cutensor.jl")
end

# note: Bumper should not be loaded before this point
# as there is a test which requires it to be loaded after
@testset "Bumper extension" verbose = true begin
    include("butensor.jl")
end

# note: OMEinsumContractionOrders should not be loaded before this point
# as there is a test which requires it to be loaded after
# the tests only work when extension is supported (julia version >= 1.9)
if isdefined(Base, :get_extension)
    @testset "OMEinsumOptimizer extension" begin
        include("omeinsum.jl")
    end
end

@testset "Polynomials" begin
    include("polynomials.jl")
end

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(TensorOperations)
end
