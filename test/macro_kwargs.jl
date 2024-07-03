using TensorOperations
using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
using Test
using Logging

@testset "backend" begin
    D1, D2, D3 = 30, 40, 20
    d1, d2 = 2, 3
    A1 = randn(D1, d1, D2) .- 1 // 2
    A2 = randn(D2, d2, D3) .- 1 // 2
    rhoL = randn(D1, D1) .- 1 // 2
    rhoR = randn(D3, D3) .- 1 // 2
    H = randn(d1, d2, d1, d2) .- 1 // 2

    E1 = @tensor backend = StridedNative() begin
        tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] *
                     H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c']))
    end
    E2 = @tensor backend = StridedBLAS() begin
        tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] *
                     H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c']))
    end
    E3 = @tensor backend = BaseView() begin
        tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] *
                     H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c']))
    end
    E4 = @tensor backend = BaseCopy() begin
        tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] *
                     H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c']))
    end
    @test E1 ≈ E2
    @test E1 ≈ E3
    @test E1 ≈ E4
end

@testset "contractcheck" begin
    A = randn(2, 2, 2)
    B = randn(2, 2, 2)
    @tensor C1[i, j, k, l] := A[i, j, m] * B[k, l, m]
    @tensor contractcheck = true C2[i, j, k, l] := A[i, j, m] * B[k, l, m]
    @test C1 ≈ C2
    B = rand(2, 2, 3)
    @test_throws DimensionMismatch("Nonmatching dimensions for m: 2 != 3") begin
        @tensor contractcheck = true C[i, j, k, l] := A[i, j, m] * B[k, l, m]
    end
end

@testset "costcheck" begin
    d, D, V = 4, 24, 2
    A = randn(D, d, D)
    ρL = randn(D, V, D)
    O = randn(V, d, d, V)
    ρR = randn(D, V, D)

    @testset "warn" begin
        E1 = @test_logs (:warn,) begin
            @tensor costcheck = warn begin
                A[1 2; 6] * ρL[5 3; 1] * O[3 4; 2 7] * ρR[6 7; 8] * conj(A[5 4; 8])
            end
        end

        # no more warning when fixing the order
        E2 = @test_logs min_level = Logging.Warn begin
            @tensor costcheck = warn order = (8, 1, 5, 6, 3, 4, 2, 7) begin
                A[1 2; 6] * ρL[5 3; 1] * O[3 4; 2 7] * ρR[6 7; 8] * conj(A[5 4; 8])
            end
        end

        @test E1 ≈ E2
    end

    @testset "cache" begin
        empty!(TensorOperations.costcache)
        E1 = @tensor costcheck = cache begin
            A[1 2; 6] * ρL[5 3; 1] * O[3 4; 2 7] * ρR[6 7; 8] * conj(A[5 4; 8])
        end
        @test !isempty(TensorOperations.costcache)
        empty!(TensorOperations.costcache)
        E2 = @tensor costcheck = warn order = (8, 1, 5, 6, 3, 4, 2, 7) begin
            A[1 2; 6] * ρL[5 3; 1] * O[3 4; 2 7] * ρR[6 7; 8] * conj(A[5 4; 8])
        end
        @test isempty(TensorOperations.costcache)
        @test E1 ≈ E2
    end
end

@testset "opt" begin
    A = randn(5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)

    @tensor opt = (a => χ, b => χ^2, c => 2 * χ, d => χ, e => 5, f => 2 * χ) begin
        D1[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensor opt = (a=χ, b=χ^2, c=2 * χ, d=χ, e=5, f=2 * χ) begin
        D2[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensor opt = ((a, d) => χ, b => χ^2, (c, f) => 2 * χ, e => 5) begin
        D3[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensor opt = ((a, d)=χ, b=χ^2, (c, f)=2 * χ, e=5) begin
        D4[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensor opt = true begin
        D5[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @test D1 ≈ D2 ≈ D3 ≈ D4 ≈ D5
end
