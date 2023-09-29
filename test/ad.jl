using TensorOperations
using Test
using ChainRulesTestUtils

ChainRulesTestUtils.test_method_tables()

precision(::Type{<:Union{Float32,Complex{Float32}}}) = 1e-2
precision(::Type{<:Union{Float64,Complex{Float64}}}) = 1e-8

@testset "tensortrace! ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    pC = ((3, 5, 2), ())
    pA = ((1,), (4,))
    α = rand(T)
    β = rand(T)
    A = rand(T, (2, 3, 4, 2, 5))
    C = rand(T, size.(Ref(A), pC[1]))
    test_rrule(tensortrace!, C, pC, A, pA, :N, α, β; atol, rtol)
end

@testset "tensoradd! ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    p = ((2, 1, 4, 3, 5), ())
    A = rand(T, (2, 3, 4, 2, 1))
    C = rand(T, size.(Ref(A), p[1]))
    α = rand(T)
    β = rand(T)
    test_rrule(tensoradd!, C, p, A, :N, α, β; atol, rtol)
    test_rrule(tensoradd!, C, p, A, :C, α, β; atol, rtol)
end

@testset "tensorcontract! ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    pC = ((3, 2, 4, 1), ())
    pA = ((2, 4, 5), (1, 3))
    pB = ((2, 1), (3,))

    A = rand(T, (2, 3, 4, 2, 5))
    B = rand(T, (4, 2, 3))
    C = rand(T, (5, 2, 3, 3))
    α = randn(T)
    β = randn(T)

    test_rrule(tensorcontract!, C, pC, A, pA, :N, B, pB, :N, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, pC, A, pA, :C, B, pB, :N, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, pC, A, pA, :N, B, pB, :C, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, pC, A, pA, :C, B, pB, :C, α, β; atol, rtol)
end

@testset "tensorscalar ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    C = Array{T,0}(undef, ())
    fill!(C, rand(T))
    test_rrule(tensorscalar, C; atol, rtol)
end
