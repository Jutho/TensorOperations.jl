# test index notation using @tensor macro
#-----------------------------------------
withblas = TensorOperations.use_blas() ? "with" : "without"
withcache = TensorOperations.use_cache() ? "with" : "without"
@testset "Index Notation $withblas BLAS and $withcache cache" begin
    t0 = time()
    A = randn(Float64, (3,5,4,6))
    p = (4,1,3,2)
    C1 = permutedims(A, p)
    @tensor C2[4,1,3,2] := A[1,2,3,4]
    @test C1 ≈ C2
    @test_throws TensorOperations.IndexError begin
        @tensor C[1,2,3,4] := A[1,2,3]
    end
    @test_throws TensorOperations.IndexError begin
        @tensor C[1,2,3,4] := A[1,2,2,4]
    end
    println("tensorcopy: $(time()-t0) seconds")
    t0 = time()

    B=randn(Float64, (5,6,3,4))
    p=[3,1,4,2]
    @tensor C1[3,1,4,2] := A[3,1,4,2] + B[1,2,3,4]
    C2=A+permutedims(B, p)
    @test C1 ≈ C2
    @test_throws DimensionMismatch begin
        @tensor C[1,2,3,4] := A[1,2,3,4] + B[1,2,3,4]
    end
    println("tensoradd: $(time()-t0) seconds")
    t0 = time()

    A=randn(Float64, (50,100,100))
    @tensor C1[a] := A[a, b', b']
    C2=zeros(50)
    for i=1:50
        for j=1:100
            C2[i]+=A[i, j, j]
        end
    end
    @test C1 ≈ C2
    A=randn(Float64, (3,20,5,3,20,4,5))
    @tensor C1[e, a, d] := A[a, b, c, d, b, e, c]
    C2=zeros(4,3,3)
    for i1=1:4, i2=1:3, i3=1:3
        for j1=1:20, j2=1:5
            C2[i1, i2, i3]+=A[i2, j1, j2, i3, j1, i1, j2]
        end
    end
    @test C1 ≈ C2
    println("tensortrace: $(time()-t0) seconds")
    t0 = time()

    A=randn(Float64, (3,20,5,3,4))
    B=randn(Float64, (5,6,20,3))
    @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
    C2=zeros(3,3,4,3,6)
    for a=1:3, b=1:20, c=1:5, d=1:3, e=1:4, f=1:6, g=1:3
        C2[a, g, e, d, f] += A[a, b, c, d, e] * B[c, f, b, g]
    end
    @test C1 ≈ C2
    @test_throws TensorOperations.IndexError begin
        @tensor A[a, b, c, d] * B[c, f, b, g]
    end
    println("tensorcontract 1: $(time()-t0) seconds")
    t0 = time()

    A=randn(Float64, (5,5,5,5))
    B=rand(ComplexF64, (5,5,5,5))
    @tensor C1[1,2,5,6,3,4,7,8] := A[1,2,3,4] * B[5,6,7,8]
    C2=reshape(kron(reshape(B, (25,25)), reshape(A, (25,25))), (5,5,5,5,5,5,5,5))
    @test C1 ≈ C2
    @test_throws TensorOperations.IndexError begin
        @tensor C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
    end
    println("tensorcontract 2: $(time()-t0) seconds")
    t0 = time()

    Da=10
    Db=15
    Dc=4
    Dd=8
    De=6
    Df=7
    Dg=3
    Dh=2
    A=rand(ComplexF64, (Da, Dc, Df, Da, De, Db, Db, Dg))
    B=rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
    C=rand(ComplexF64, (Dd, Dh, Df))
    @tensor D1[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d]+0.5 * C[d, h, f]
    D2=zeros(ComplexF64, (Dd, Df, Dh))
    for d=1:Dd, f=1:Df, h=1:Dh
        D2[d, f, h] += 0.5 * C[d, h, f]
        for a=1:Da, b=1:Db, c=1:Dc, e=1:De, g=1:Dg
            D2[d, f, h] += A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d]
        end
    end
    @test D1 ≈ D2
    @test norm(vec(D1)) ≈ sqrt(abs((@tensor scalar(D1[d, f, h] * conj(D1[d, f, h])))))
    println("tensorcontract 3: $(time()-t0) seconds")
    t0 = time()


    Abig=randn(Float64, (30,30,30,30))
    A=view(Abig,1 .+ 3 .* (0:9),2 .+ 2 .* (0:6),5 .+ 4 .* (0:6),4 .+ 3 .* (0:8))
    p=[3,1,4,2]
    Cbig=zeros(ComplexF64, (50,50,50,50))
    C=view(Cbig,13 .+ (0:6),11 .+ 4 .* (0:9),15 .+ 4 .* (0:8),4 .+ 3 .* (0:6))
    Acopy = copy(A)
    Ccopy = copy(C)
    @tensor C[3,1,4,2] = A[1,2,3,4]
    @tensor Ccopy[3,1,4,2] = Acopy[1,2,3,4]
    @test C ≈ Ccopy
    @test_throws TensorOperations.IndexError begin
        @tensor C[3,1,4,2] = A[1,2,3]
    end
    @test_throws DimensionMismatch begin
        @tensor C[3,1,4,2] = A[3,1,4,2]
    end
    @test_throws TensorOperations.IndexError begin
        @tensor C[1,1,2,3] = A[1,2,3,4]
    end
    println("views: $(time()-t0) seconds")
    t0 = time()

    Cbig=zeros(ComplexF64, (50,50,50,50))
    C=view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
    Acopy=permutedims(copy(A), p)
    Ccopy=copy(C)
    α=randn(Float64)
    β=randn(Float64)
    @tensor C[3,1,4,2] = β * C[3,1,4,2] + α * A[1,2,3,4]
    Ccopy=β * Ccopy+α * Acopy
    @test C ≈ Ccopy
    @test_throws TensorOperations.IndexError begin
        @tensor C[3,1,4,2] = 0.5 * C[3,1,4,2] + 1.2 * A[1,2,3]
    end
    @test_throws DimensionMismatch  begin
        @tensor C[3,1,4,2] = 0.5 * C[3,1,4,2] + 1.2 * A[3,1,2,4]
    end
    @test_throws TensorOperations.IndexError  begin
        @tensor C[1,1,2,3] = 0.5 * C[1,1,2,3] + 1.2 * A[1,2,3,4]
    end
    println("more views: $(time()-t0) seconds")
    t0 = time()

    Abig=rand(Float64, (30,30,30,30))
    A=view(Abig,1 .+ 3 .* (0:8),2 .+ 2 .* (0:14),5 .+ 4 .* (0:6),7 .+ 2 .* (0:8))
    Bbig=rand(ComplexF64, (50,50))
    B=view(Bbig,13 .+ (0:14),3 .+ 5 .* (0:6))
    Acopy=copy(A)
    Bcopy=copy(B)
    α=randn(Float64)
    @tensor B[b, c] += α * A[a, b, c, a]
    for i=1 .+ (0:8)
        Bcopy += α * view(A, i, :, :, i)
    end
    @test B ≈ Bcopy
    @test_throws TensorOperations.IndexError begin
        @tensor B[b, c] += α * A[a, b, c]
    end
    @test_throws DimensionMismatch begin
        @tensor B[c, b] += α * A[a, b, c, a]
    end
    @test_throws TensorOperations.IndexError begin
        @tensor B[c, b] += α * A[a, b, a, a]
    end
    @test_throws DimensionMismatch begin
        @tensor B[c, b] += α * A[a, b, a, c]
    end
    println("even more views: $(time()-t0) seconds")
    t0 = time()

    Abig=rand(Float64, (30,30,30,30))
    A=view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
    Bbig=rand(ComplexF64, (50,50,50))
    B=view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
    Cbig=rand(ComplexF32, (40,40,40))
    C=view(Cbig,3 .+ 2 .* (0:8),13 .+ (0:8),7 .+ 3 .* (0:7))
    Acopy=copy(A)
    Bcopy=copy(B)
    Ccopy=copy(C)
    α=randn(Float64)
    for d=1 .+ (0:8), a=1 .+ (0:8), e=1 .+ (0:7)
        for b=1 .+ (0:14), c=1 .+ (0:6)
            Ccopy[d, a, e] -=α * A[a, b, c, d] * conj(B[c, e, b])
        end
    end
    @tensor C[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
    @test C ≈ Ccopy
    println("and some more views: $(time()-t0) seconds")
    t0 = time()

    Cbig=rand(ComplexF32, (40,40,40))
    C=view(Cbig,3 .+ 2 .* (0:8),13 .+ (0:8),7 .+ 3 .* (0:7))
    Ccopy=copy(C)
    for d=1 .+ (0:8), a=1 .+ (0:8), e=1 .+ (0:7)
        for b=1 .+ (0:14), c=1 .+ (0:6)
            Ccopy[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
        end
    end
    @tensor C[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
    @test C ≈ Ccopy
    @test_throws TensorOperations.IndexError begin
        @tensor C[d, a, e] += α * A[a, b, c, a] * B[c, e, b]
    end
    @test_throws TensorOperations.IndexError begin
        @tensor C[d, a, e] += α * A[a, b, c, d] * B[c, b]
    end
    @test_throws TensorOperations.IndexError begin
        @tensor C[d, e] += α * A[a, b, c, d] * B[c, e, b]
    end
    @test_throws DimensionMismatch begin
        @tensor C[d, e, a] += α * A[a, b, c, d] * B[c, e, b]
    end
    println("Float32 views: $(time()-t0) seconds")
    t0 = time()

    # Example from README.md
    using TensorOperations
    α=randn()
    A=randn(5,5,5,5,5,5)
    B=randn(5,5,5)
    C=randn(5,5,5)
    D=zeros(5,5,5)
    @tensor begin
        D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    end
    @test D == E
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
        rA12 = reshape(reshape(rhoL * reshape(A12, (D1, d1*d2*D3)), (D1*d1*d2, D3)) * rhoR, (D1, d1, d2, D3))
        HrA12 = permutedims(reshape(reshape(H, (d1 * d2, d1*d2)) * reshape(permutedims(rA12, (2,3,1,4)), (d1 * d2, D1 * D3)), (d1, d2, D1, D3)), (3,1,2,4))
        E = dot(A12, HrA12)
        @tensor HrA12′[a, s1, s2, c] := rhoL[a, a'] * A1[a', t1, b] * A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]
        @tensor HrA12′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] * A2[2, 5, 3] * rhoR[3, -4] * A1[1, 4, 2] # should be contracted in exactly same order
        @test HrA12′ == HrA12′′ # should be exactly equal
        @test HrA12 ≈ HrA12′
        @test E ≈ @tensor scalar(rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] * H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c']))
    end
    println("tensor network examples: $(time()-t0) seconds")
    t0 = time()

    # diagonal matrices
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = randn(T, 10, 10, 10, 10)
        @tensor C[3,1,4,2] := A[1,2,3,4]
        U,S,V = svd(reshape(C, (100, 100)))
        U2 = reshape(view(U, 1:2:100, :), (5,10,100))
        S2 = Diagonal(S)
        V2 = reshape(view(V, 1:2:100, :), (5, 10, 100))
        @tensor A2[1,2,3,4] := U2[3,1,a]*S2[a,a']*conj(V2[4,2,a'])
        @test A2 ≈ view(A, :, :, 1:2:10, 1:2:10)

        B = randn(T, 10)
        @tensor C1[3,1,4,2] := A[a,2,3,4]*Diagonal(B)[a,1]
        @tensor C2[3,1,4,2] := A[1,a,3,4]*conj(Diagonal(B)[a,2])
        @tensor C3[3,1,4,2] := conj(A[1,2,a,4])*Diagonal(B)[a,3]
        @tensor C4[3,1,4,2] := conj(A[1,2,3,a])*conj(Diagonal(B)[a,4])
        @tensor C1′[3,1,4,2] := Diagonal(B)[a,1]*A[a,2,3,4]
        @tensor C2′[3,1,4,2] := conj(Diagonal(B)[a,2])*A[1,a,3,4]
        @tensor C3′[3,1,4,2] := Diagonal(B)[a,3]*conj(A[1,2,a,4])
        @tensor C4′[3,1,4,2] := conj(Diagonal(B)[a,4])*conj(A[1,2,3,a])
        @test C1 == C1′
        @test C2 == C2′
        @test C3 == C3′
        @test C4 == C4′
        for i = 1:10
            @test C1[:,i,:,:] ≈ permutedims(A[i,:,:,:],(2,3,1))*B[i]
            @test C2[:,:,:,i] ≈ permutedims(A[:,i,:,:],(2,1,3))*conj(B[i])
            @test C3[i,:,:,:] ≈ conj(permutedims(A[:,:,i,:],(1,3,2)))*B[i]
            @test C4[:,:,i,:] ≈ conj(permutedims(A[:,:,:,i],(3,1,2)))*conj(B[i])
        end

        @tensor D1[1,2] := A[1,2,a,b]*Diagonal(B)[a,b]
        @tensor D2[1,2] := A[1,b,2,a]*conj(Diagonal(B)[a,b])
        @tensor D3[1,2] := conj(A[a,2,1,b])*Diagonal(B)[a,b]
        @tensor D4[1,2] := conj(A[a,1,b,2])*conj(Diagonal(B)[a,b])
        @tensor D1′[1,2] := Diagonal(B)[a,b]*A[1,2,a,b]
        @tensor D2′[1,2] := conj(Diagonal(B)[a,b])*A[1,b,2,a]
        @tensor D3′[1,2] := Diagonal(B)[a,b]*conj(A[a,2,1,b])
        @tensor D4′[1,2] := conj(Diagonal(B)[a,b])*conj(A[a,1,b,2])
        @test D1 == D1′
        @test D2 == D2′
        @test D3 == D3′
        @test D4 == D4′

        E1 = zero(D1)
        E2 = zero(D2)
        E3 = zero(D3)
        E4 = zero(D4)
        for i = 1:10
            E1[:,:] += A[:,:,i,i]*B[i]
            E2[:,:] += A[:,i,:,i]*conj(B[i])
            E3[:,:] += A[i,:,:,i]'*B[i]
            E4[:,:] += conj(A[i,:,i,:])*conj(B[i])
        end
        @test D1 ≈ E1
        @test D2 ≈ E2
        @test D3 ≈ E3
        @test D4 ≈ E4

        F = randn(T,(10,10))
        @tensor G[a,c,b,d] := F[a,b]*Diagonal(B)[c,d]
        @test reshape(G,(100,100)) ≈ kron(Diagonal(B), F)
    end
    println("diagonal examples: $(time()-t0) seconds")
    t0 = time()


end
