using TensorOperations
using Base.Test

if VERSION.minor < 3
    vecnorm(x) = norm(vec(x))
    include("tests2.jl")
else
    include("tests3.jl")
end
