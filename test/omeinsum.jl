@testset "@tensor dependency check" begin
    A = rand(2, 2)
    B = rand(2, 2)
    C = rand(2, 2)
    @test_throws ArgumentError begin
        ex = :(@tensor opt = (i => 2, j => 2, k => 2) opt_algorithm = GreedyMethod S[] := A[i,
                                                                                            j] *
                                                                                          B[j,
                                                                                            k] *
                                                                                          C[i,
                                                                                            k])
        macroexpand(Main, ex)
    end
end

using OMEinsumContractionOrders


@testset "OMEinsumContractionOrders optimization algorithms" begin
    A = randn(5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)

    @tensor begin
        D1[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (a => 5, b => 5, c => 5, d => 5, e => 5, f => 5, g => 5) opt_algorithm = GreedyMethod begin
        D2[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (a => 5, b => 5, c => 5, d => 5, e => 5, f => 5, g => 5) opt_algorithm = TreeSA begin
        D3[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (a => 5, b => 5, c => 5, d => 5, e => 5, f => 5, g => 5) opt_algorithm = KaHyParBipartite begin
        D4[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (a => 5, b => 5, c => 5, d => 5, e => 5, f => 5, g => 5) opt_algorithm = SABipartite begin
        D5[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (a => 5, b => 5, c => 5, d => 5, e => 5, f => 5, g => 5) opt_algorithm = ExactTreewidth begin
        D6[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @tensor opt = (1 => 5, 2 => 5, 3 => 5, 4 => 5, 5 => 5, 6 => 5, 7 => 5) opt_algorithm = GreedyMethod begin
        D7[1, 2, 3, 4] := A[1, 5, 3, 6] * B[7, 4, 5] * C[7, 6, 2]
    end

    # check the case that opt_algorithm is before the opt
    @tensor opt_algorithm = GreedyMethod opt = (a => 5, b => 5, c => 5, d => 5, e => 5,
                                                f => 5, g => 5) begin
        D8[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    @test D1 ≈ D2 ≈ D3 ≈ D4 ≈ D5 ≈ D6 ≈ D7 ≈ D8

    A = rand(2, 2)
    B = rand(2, 2, 2)
    C = rand(2, 2)
    D = rand(2, 2)
    E = rand(2, 2, 2)
    F = rand(2, 2)

    @tensor opt = true begin
        s1[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @tensor opt = (i => 2, j => 2, k => 2, l => 2, m => 2, n => 2, o => 2) opt_algorithm = GreedyMethod begin
        s2[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @tensor opt = (i => 2, j => 2, k => 2, l => 2, m => 2, n => 2, o => 2) opt_algorithm = TreeSA begin
        s3[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @tensor opt = (i => 2, j => 2, k => 2, l => 2, m => 2, n => 2, o => 2) opt_algorithm = KaHyParBipartite begin
        s4[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @tensor opt = (i => 2, j => 2, k => 2, l => 2, m => 2, n => 2, o => 2) opt_algorithm = SABipartite begin
        s5[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @tensor opt = (i => 2, j => 2, k => 2, l => 2, m => 2, n => 2, o => 2) opt_algorithm = ExactTreewidth begin
        s6[] := A[i, k] * B[i, j, l] * C[j, m] * D[k, n] * E[n, l, o] * F[o, m]
    end

    @test s1 ≈ s2 ≈ s3 ≈ s4 ≈ s5 ≈ s6

    A = randn(5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)
    α = randn()

    @tensor opt = true begin
        D1[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @tensor opt = (i => 5, j => 5, k => 5, l => 5, m => 5) opt_algorithm = GreedyMethod begin
        D2[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @tensor opt = (i => 5, j => 5, k => 5, l => 5, m => 5) opt_algorithm = TreeSA begin
        D3[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @tensor opt = (i => 5, j => 5, k => 5, l => 5, m => 5) opt_algorithm = KaHyParBipartite begin
        D4[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @tensor opt = (i => 5, j => 5, k => 5, l => 5, m => 5) opt_algorithm = SABipartite begin
        D5[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @tensor opt = (i => 5, j => 5, k => 5, l => 5, m => 5) opt_algorithm = ExactTreewidth begin
        D6[m] := A[i, j, k] * B[j, k, l] * C[i, l, m] +
                 α * A[i, j, k] * B[j, k, l] * C[i, l, m]
    end

    @test D1 ≈ D2 ≈ D3 ≈ D4 ≈ D5 ≈ D6
end

@testset "ncon with OMEinsumContractionOrders" begin
    A = randn(5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)

    @tensor begin
        D1[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end

    D2 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]])
    D3 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = ExhaustiveSearchOptimizer())
    D4 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = GreedyMethodOptimizer())
    D5 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = KaHyParBipartiteOptimizer())
    D6 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = TreeSAOptimizer())
    D7 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = SABipartiteOptimizer())
    D8 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = ExactTreewidthOptimizer())

    @test D1 ≈ D2 ≈ D3 ≈ D4 ≈ D5 ≈ D6 ≈ D7 ≈ D8

    @test_throws ArgumentError begin
        D9 = ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], order = [5, 6, 7], optimizer = GreedyMethod())
    end

    @test_logs (:debug, "Using optimizer ExhaustiveSearch") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = ExhaustiveSearchOptimizer())

    @test_logs (:debug, "Using optimizer GreedyMethod from OMEinsumContractionOrders") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = GreedyMethodOptimizer())

    @test_logs (:debug, "Using optimizer KaHyParBipartite from OMEinsumContractionOrders") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = KaHyParBipartiteOptimizer())

    @test_logs (:debug, "Using optimizer TreeSA from OMEinsumContractionOrders") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = TreeSAOptimizer())

    @test_logs (:debug, "Using optimizer SABipartite from OMEinsumContractionOrders") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = SABipartiteOptimizer())
    
    @test_logs (:debug, "Using optimizer ExactTreewidth from OMEinsumContractionOrders") min_level=Logging.Debug match_mode=:any ncon([A, B, C], [[-1, 5, -3, 6], [7, -4, 5], [7, 6, -2]], optimizer = ExactTreewidthOptimizer())
end
