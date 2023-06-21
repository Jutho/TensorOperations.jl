using TensorOperations
using Test
using Zygote
using LinearAlgebra
using Base.Iterators: product
Zygote.refresh()

function LinAlg_tensoradd(A, pA, conjA, B, pB, conjB, α=true, β=true)
    return α * permutedims(conjA == :N ? A : conj(A), linearize(pA)) +
           β * permutedims(conjB == :N ? B : conj(B), linearize(pB))
end
function LinAlg_tensorcontract(C, pC, A, pA, conjA, B, pB, conjB, α=true, β=false)
    szA(i) = size(A, i)
    A′ = reshape(permutedims(conjA == :N ? A : conj(A), linearize(pA)), prod(szA.(pA[1])),
                 prod(szA.(pA[2])))
    szB(i) = size(B, i)
    B′ = reshape(permutedims(conjB == :N ? B : conj(B), linearize(pB)), prod(szB.(pB[1])),
                 prod(szB.(pB[2])))
    C′ = reshape(A′ * B′, szA.(pA[1])..., szB.(pB[2])...)
    return β * C + α * permutedims(C′, linearize(pC))
end
function LinAlg_tensortrace(C, pC, A, pA, conjA, α=true, β=false)
    szA(i) = size(A, i)
    A′ = reshape(permutedims(conjA == :N ? A : conj(A),
                             (linearize(pC)..., pA[1]..., pA[2]...)),
                 prod(szA.(linearize(pC))), prod(szA.(pA[1])), prod(szA.(pA[2])))
    C′ = map(i -> tr(A′[i, :, :]), axes(A′, 1))
    return β * C + α * reshape(C′, szA.(linearize(pC)))
end

precision(T::Type{<:Complex}) = precision(real(T))
precision(T::Type{<:Number}) = eps(T)^(3 / 4)

@testset "tensoradd" begin
    f(A, B) = tensoradd(A, ((1, 2, 3), ()), :N, B, ((1, 3, 2), ()), :N)
    f′(A, B) = LinAlg_tensoradd(A, ((1, 2, 3), ()), :N, B, ((1, 3, 2), ()), :N)

    @testset for T in (Float64, ComplexF64)
        A = rand(T, 2, 3, 4)
        B = rand(T, 2, 4, 3)

        C, pullback = Zygote.pullback(f, A, B)
        C′, pullback′ = Zygote.pullback(f′, A, B)

        @test C ≈ C′ rtol = precision(T)

        ΔC = rand(T, size(C))
        ΔA, ΔB = pullback(ΔC)
        ΔA′, ΔB′ = pullback′(ΔC)
        @test ΔA ≈ ΔA′ rtol = precision(T)
        @test ΔB ≈ ΔB′ rtol = precision(T)

        D = rand(T, 4, 2, 3, 2)
        E = rand(T, 2, 3, 4, 2)
        α = rand(T)
        β = rand(T)

        pD = ((2, 1, 4, 3), ())
        pE = ((1, 3, 4, 2), ())

        for conjD in (:N, :C), conjE in (:N, :C)
            F, pullback2 = Zygote.pullback(tensoradd, D, pD, conjD, E, pE, conjE, α, β)
            F′, pullback2′ = Zygote.pullback(LinAlg_tensoradd, D, pD, conjD, E, pE, conjE,
                                             α, β)
            @test F ≈ F′ rtol = precision(T)

            ΔF = rand(T, size(F))
            ΔD, ΔpD, ΔconjD, ΔE, ΔpE, ΔconjE, Δα, Δβ = pullback2(ΔF)
            ΔD′, ΔpD′, ΔconjD′, ΔE′, ΔpE′, ΔconjE′, Δα′, Δβ′ = pullback2′(ΔF)
            @test ΔD ≈ ΔD′ rtol = precision(T)
            @test ΔE ≈ ΔE′ rtol = precision(T)
            @test Δα ≈ Δα′ rtol = precision(T)
            @test Δβ ≈ Δβ′ rtol = precision(T)
        end
    end
end

@testset "tensorcontract" begin
    @testset for T in (Float64, ComplexF64)
        A = rand(T, 2, 4, 3, 2)
        B = rand(T, 1, 3, 2)
        C = rand(T, 1, 4, 2)
        
        α = rand(T)
        β = rand(T)

        pA = ((2, 4), (1, 3))
        pB = ((3, 2), (1,))
        pC = ((3, 1, 2), ())

        for conjA in (:N, :C), conjB in (:N, :C)
            D, pullback = Zygote.pullback(tensorcontract!, C, pC, A, pA, conjA, B, pB,
                                          conjB, α,
                                          β)
            D′, pullback′ = Zygote.pullback(LinAlg_tensorcontract, C, pC, A, pA, conjA, B,
                                            pB,
                                            conjB, α, β)

            @test D ≈ D′ rtol = precision(T)
            ΔD = rand(T, size(D))
            ΔC, ΔpC, ΔA, ΔpA, ΔconjA, ΔB, ΔpB, ΔconjB, Δα, Δβ = pullback(ΔD)
            ΔC′, ΔpC′, ΔA′, ΔpA′, ΔconjA′, ΔB′, ΔpB′, ΔconjB′, Δα′, Δβ′ = pullback′(ΔD)
            @test ΔC ≈ ΔC′ rtol = precision(T)
            @test ΔA ≈ ΔA′ rtol = precision(T)
            @test ΔB ≈ ΔB′ rtol = precision(T)
            @test Δα ≈ Δα′ rtol = precision(T)
            @test Δβ ≈ Δβ′ rtol = precision(T)
        end
    end
end

@testset "tensortrace" begin
    @testset for T in (Float64, ComplexF64)
        A = rand(T, 2, 3, 4, 2)
        C = rand(T, 4, 3)
        α = rand(T)
        β = rand(T)

        pA = ((1,), (4,))
        pC = ((3, 2), ())

        conjA = :N

        D, pullback = Zygote.pullback(tensortrace!, C, pC, A, pA, conjA, α, β)
        D′, pullback′ = Zygote.pullback(LinAlg_tensortrace, C, pC, A, pA, conjA, α, β)
        @test D ≈ D′ rtol = precision(T)

        ΔD = rand(T, size(D))
        ΔC, ΔpC, ΔA, ΔpA, ΔconjA, Δα, Δβ = pullback(ΔD)
        ΔC′, ΔpC′, ΔA′, ΔpA′, ΔconjA′, Δα′, Δβ′ = pullback′(ΔD)
        @test ΔC ≈ ΔC′ rtol = precision(T)
        @test ΔA ≈ ΔA′ rtol = precision(T)
        @test Δα ≈ Δα′ rtol = precision(T)
        @test Δβ ≈ Δβ′ rtol = precision(T)
    end
end
