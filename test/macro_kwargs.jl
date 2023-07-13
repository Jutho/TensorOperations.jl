using TensorOperations, Test

@testset "opt" begin
end

@testset "contractcheck" begin
    A = randn(2, 2, 2)
    B = randn(2, 2, 3)
    @test_throws DimensionMismatch("Nonmatching dimensions for j: 2 != 3") begin
        @tensor contractcheck = true C[i, j, k] := A[i, j, l] * B[l, k, j]
    end
end

@testset "costcheck" begin end

@testset "backend" begin end
