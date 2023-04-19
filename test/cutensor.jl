using TensorOperations
using Test
using LinearAlgebra

@testset "cutensor macro for elementary operations" begin
    t0 = time()
    A = randn(Float64, (3, 5, 4, 6))
    p = (4, 1, 3, 2)
    @cutensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
    C2 = permutedims(A, p)
    @test collect(C1) ≈ C2
    @test_throws TensorOperations.IndexError begin
        @cutensor C[1, 2, 3, 4] := A[1, 2, 3]
    end
    @test_throws TensorOperations.IndexError begin
        @cutensor C[1, 2, 3, 4] := A[1, 2, 2, 4]
    end
    println("tensorcopy: $(time()-t0) seconds")
    t0 = time()

    B = randn(Float64, (5, 6, 3, 4))
    p = [3, 1, 4, 2]
    α = randn(Float64)
    @cutensor C1[3, 1, 4, 2] := A[3, 1, 4, 2] + α * B[1, 2, 3, 4] # mixed eltype add/permutation
    C2 = A + α * permutedims(B, p)
    @test collect(C1) ≈ C2
    @test_throws cuTENSOR.CUTENSORError begin
        @cutensor C[1, 2, 3, 4] := A[1, 2, 3, 4] + B[1, 2, 3, 4]
    end
    println("tensoradd: $(time()-t0) seconds")
    t0 = time()

    A = randn(Float64, (50, 100, 100))
    @cutensor C1[a] := A[a, b', b']
    C2 = zeros(50)
    for i in 1:50
        for j in 1:100
            C2[i] += A[i, j, j]
        end
    end
    @test collect(C1) ≈ C2
    A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
    @cutensor C1[e, a, d] := A[a, b, c, d, b, e, c]
    C2 = zeros(4, 3, 3)
    for i1 in 1:4, i2 in 1:3, i3 in 1:3
        for j1 in 1:20, j2 in 1:5
            C2[i1, i2, i3] += A[i2, j1, j2, i3, j1, i1, j2]
        end
    end
    @test collect(C1) ≈ C2
    println("tensortrace: $(time()-t0) seconds")
    t0 = time()

    A = randn(Float64, (3, 20, 5, 3, 4))
    B = randn(Float64, (5, 6, 20, 3))
    @cutensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
    C2 = zeros(3, 3, 4, 3, 6)
    for a in 1:3, b in 1:20, c in 1:5, d in 1:3, e in 1:4, f in 1:6, g in 1:3
        C2[a, g, e, d, f] += A[a, b, c, d, e] * B[c, f, b, g]
    end
    @test collect(C1) ≈ C2
    @test_throws TensorOperations.IndexError begin
        @cutensor A[a, b, c, d] * B[c, f, b, g]
    end
    println("tensorcontract 1: $(time()-t0) seconds")
    t0 = time()

    A = randn(Float64, (5, 5, 5, 5))
    B = rand(ComplexF64, (5, 5, 5, 5))
    # @cutensor C1[1,2,5,6,3,4,7,8] := A[1,2,3,4] * B[5,6,7,8]
    # C2=reshape(kron(reshape(B, (25,25)), reshape(A, (25,25))), (5,5,5,5,5,5,5,5))
    # @test C1 ≈ C2
    @test_throws TensorOperations.IndexError begin
        @cutensor C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
    end
    println("tensorcontract 2: $(time()-t0) seconds")
    t0 = time()
end

@testset "cutensor macro for more complicated expressions" begin
    t0 = time()
    Da = 10
    Db = 15
    Dc = 4
    Dd = 8
    De = 6
    Df = 7
    Dg = 3
    Dh = 2
    A = rand(ComplexF64, (Dc, Da, Df, Da, De, Db, Db, Dg))
    B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
    C = rand(ComplexF64, (Dd, Dh, Df))
    @cutensor D1[d, f, h] := A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d] + 0.5 * C[d, h, f]
    D2 = zeros(ComplexF64, (Dd, Df, Dh))
    for d in 1:Dd, f in 1:Df, h in 1:Dh
        D2[d, f, h] += 0.5 * C[d, h, f]
        for a in 1:Da, b in 1:Db, c in 1:Dc, e in 1:De, g in 1:Dg
            D2[d, f, h] += A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d]
        end
    end
    @test collect(D1) ≈ D2
    @test norm(vec(D1)) ≈
          sqrt(abs((@cutensor tensorscalar(D1[d, f, h] * conj(D1[d, f, h])))))
    println("tensorcontract 3: $(time()-t0) seconds")
    t0 = time()

    # Example from README.md
    α = randn()
    A = randn(5, 5, 5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)
    D = zeros(5, 5, 5)
    @cutensor begin
        D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    end
    @test D == collect(E)
    println("readme example: $(time()-t0) seconds")
    t0 = time()

    # Some tensor network examples
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        D1, D2, D3 = 30, 40, 20
        d1, d2 = 2, 3
        A1 = randn(T, D1, d1, D2)
        A2 = randn(T, D2, d2, D3)
        rhoL = randn(T, D1, D1)
        rhoR = randn(T, D3, D3)
        H = randn(T, d1, d2, d1, d2)
        A12 = reshape(reshape(A1, D1 * d1, D2) * reshape(A2, D2, d2 * D3), (D1, d1, d2, D3))
        rA12 = reshape(reshape(rhoL * reshape(A12, (D1, d1 * d2 * D3)),
                               (D1 * d1 * d2, D3)) * rhoR, (D1, d1, d2, D3))
        HrA12 = permutedims(reshape(reshape(H, (d1 * d2, d1 * d2)) *
                                    reshape(permutedims(rA12, (2, 3, 1, 4)),
                                            (d1 * d2, D1 * D3)), (d1, d2, D1, D3)),
                            (3, 1, 2, 4))
        E = dot(A12, HrA12)
        @cutensor HrA12′[a, s1, s2, c] := rhoL[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                                          rhoR[c', c] * H[s1, s2, t1, t2]
        @cutensor HrA12′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] * A2[2, 5, 3] * rhoR[3, -4] *
                                A1[1, 4, 2] # should be contracted in exactly same order
        @test collect(HrA12′) == collect(HrA12′′) # should be exactly equal
        @test HrA12 ≈ collect(HrA12′)
        @test HrA12 ≈ collect(HrA12′′)
        @test E ≈ @cutensor tensorscalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] *
                                         rhoR[c, c'] * H[t, t', s, s'] * conj(A1[a', t, b']) *
                                         conj(A2[b', t', c']))
    end
    println("tensor network examples: $(time()-t0) seconds")
    t0 = time()
end
