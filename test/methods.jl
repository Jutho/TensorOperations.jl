# test simple methods
#---------------------
withblas = TensorOperations.use_blas() ? "with" : "without"
withcache = TensorOperations.use_cache() ? "with" : "without"
@testset "Method syntax $withblas BLAS and $withcache cache" begin
    @testset "tensorcopy" begin
        A = randn(Float64, (3,5,4,6))
        p = randperm(4)
        C1 = permutedims(A,p)
        C2 = @inferred tensorcopy(A,(1:4...,),(p...,))
        @test C1 ≈ C2
        @test_throws TensorOperations.IndexError tensorcopy(A,1:3,1:4)
        @test_throws TensorOperations.IndexError tensorcopy(A,[1,2,2,4],1:4)
    end

    @testset "tensoradd" begin
        A = randn(Float64, (3,5,4,6))
        B = randn(Float64, (5,6,3,4))
        p = (3,1,4,2)
        C1 = @inferred tensoradd(A,p,B,(1:4...,))
        C2 = A+permutedims(B,p)
        @test C1 ≈ C2
        @test_throws DimensionMismatch tensoradd(A,1:4,B,1:4)
    end

    @testset "tensortrace" begin
        A = randn(Float64, (50,100,100))
        C1 = tensortrace(A,[:a,:b,:b])
        C2 = zeros(50)
        for i = 1:50
            for j = 1:100
                C2[i] += A[i,j,j]
            end
        end
        @test C1 ≈ C2
        A = randn(Float64, (3,20,5,3,20,4,5))
        C1 = @inferred tensortrace(A,(:a,:b,:c,:d,:b,:e,:c),(:e,:a,:d))
        C2 = zeros(4,3,3)
        for i1 = 1:4, i2 = 1:3, i3 = 1:3
            for j1 = 1:20,j2 = 1:5
                C2[i1,i2,i3] += A[i2,j1,j2,i3,j1,i1,j2]
            end
        end
        @test C1 ≈ C2
    end

    @testset "tensorcontract" begin
        A = randn(Float64, (3,20,5,3,4))
        B = randn(Float64, (5,6,20,3))
        C1 = @inferred tensorcontract(A,(:a,:b,:c,:d,:e),B,(:c,:f,:b,:g),(:a,:g,:e,:d,:f))
        C2 = @inferred tensorcontract(A,(:a,:b,:c,:d,:e),B,(:c,:f,:b,:g),(:a,:g,:e,:d,:f))
        C3 = zeros(3,3,4,3,6)
        for a = 1:3, b = 1:20, c = 1:5, d = 1:3, e = 1:4, f = 1:6, g = 1:3
            C3[a,g,e,d,f] += A[a,b,c,d,e]*B[c,f,b,g]
        end
        @test C1 ≈ C3
        @test C2 ≈ C3
        @test_throws TensorOperations.IndexError tensorcontract(A,[:a,:b,:c,:d],B,[:c,:f,:b,:g])
        @test_throws TensorOperations.IndexError tensorcontract(A,[:a,:b,:c,:a,:e],B,[:c,:f,:b,:g])
    end

    @testset "tensorproduct" begin
        A = randn(Float64, (5,5,5,5))
        B = rand(ComplexF64,(5,5,5,5))
        C1 = reshape((@inferred tensorproduct(A,(1,2,3,4),B,(5,6,7,8),(1,2,5,6,3,4,7,8))),(5*5*5*5,5*5*5*5))
        C2 = kron(reshape(B,(25,25)),reshape(A,(25,25)))
        @test C1 ≈ C2
        @test_throws TensorOperations.IndexError tensorproduct(A,[:a,:b,:c,:d],B,[:d,:e,:f,:g])
        @test_throws TensorOperations.IndexError tensorproduct(A,[:a,:b,:c,:d],B,[:e,:f,:g,:h],[:a,:b,:c,:d,:e,:f,:g,:i])
    end

    # test in-place methods
    #-----------------------
    # test different versions of in-place methods,
    # with changing element type and with nontrivial strides

    @testset "tensorcopy!" begin
        Abig = randn(Float64, (30,30,30,30))
        A = view(Abig,1 .+ 3*(0:9),2 .+ 2*(0:6),5 .+ 4*(0:6),4 .+ 3*(0:8))
        p = (3,1,4,2)
        Cbig = zeros(ComplexF64,(50,50,50,50))
        C = view(Cbig,13 .+ (0:6),11 .+ 4*(0:9),15 .+ 4*(0:8),4 .+ 3*(0:6))
        Acopy = tensorcopy(A,1:4,1:4)
        Ccopy = tensorcopy(C,1:4,1:4)
        TensorOperations.tensorcopy!(A,1:4,C,p)
        TensorOperations.tensorcopy!(Acopy,1:4,Ccopy,p)
        @test C ≈ Ccopy
        @test_throws TensorOperations.IndexError TensorOperations.tensorcopy!(A,1:3,C,p)
        @test_throws DimensionMismatch TensorOperations.tensorcopy!(A,p,C,p)
        @test_throws TensorOperations.IndexError TensorOperations.tensorcopy!(A,1:4,C,[1,1,2,3])
    end

    @testset "tensoradd!" begin
        Abig = randn(Float64, (30,30,30,30))
        A = view(Abig,1 .+ 3*(0:9),2 .+ 2*(0:6),5 .+ 4*(0:6),4 .+ 3*(0:8))
        p = (3,1,4,2)
        Cbig = zeros(ComplexF64,(50,50,50,50))
        C = view(Cbig,13 .+ (0:6),11 .+ 4*(0:9),15 .+ 4*(0:8),4 .+ 3*(0:6))
        Acopy = tensorcopy(A,1:4,p)
        Ccopy = tensorcopy(C,1:4,1:4)
        α = randn(Float64)
        β = randn(Float64)
        TensorOperations.tensoradd!(α,A,1:4,β,C,p)
        Ccopy = β*Ccopy+α*Acopy
        @test C ≈ Ccopy
        @test_throws TensorOperations.IndexError TensorOperations.tensoradd!(1.2,A,1:3,0.5,C,p)
        @test_throws DimensionMismatch TensorOperations.tensoradd!(1.2,A,p,0.5,C,p)
        @test_throws TensorOperations.IndexError TensorOperations.tensoradd!(1.2,A,1:4,0.5,C,[1,1,2,3])
    end

    @testset "tensortrace!" begin
        Abig = rand(Float64, (30,30,30,30))
        A = view(Abig,1 .+ 3*(0:8),2 .+ 2*(0:14),5 .+ 4*(0:6),7 .+ 2*(0:8))
        Bbig = rand(ComplexF64,(50,50))
        B = view(Bbig,13 .+ (0:14),3 .+ 5*(0:6))
        Acopy = tensorcopy(A,1:4)
        Bcopy = tensorcopy(B,1:2)
        α = randn(Float64)
        β = randn(Float64)
        TensorOperations.tensortrace!(α,A,[:a,:b,:c,:a],β,B,[:b,:c])
        Bcopy = β*Bcopy
        for i = 1 .+ (0:8)
            Bcopy += α*view(A,i,:,:,i)
        end
        @test B ≈ Bcopy
        @test_throws TensorOperations.IndexError TensorOperations.tensortrace!(α,A,[:a,:b,:c],β,B,[:b,:c])
        @test_throws DimensionMismatch TensorOperations.tensortrace!(α,A,[:a,:b,:c,:a],β,B,[:c,:b])
        @test_throws TensorOperations.IndexError TensorOperations.tensortrace!(α,A,[:a,:b,:a,:a],β,B,[:c,:b])
        @test_throws DimensionMismatch TensorOperations.tensortrace!(α,A,[:a,:b,:a,:c],β,B,[:c,:b])
    end

    @testset "tensorcontract!" begin
        Abig = rand(Float64, (30,30,30,30))
        A = view(Abig,1 .+ 3*(0:8),2 .+ 2*(0:14),5 .+ 4*(0:6),7 .+ 2*(0:8))
        Bbig = rand(ComplexF64,(50,50,50))
        B = view(Bbig,3 .+ 5*(0:6),7 .+ 2*(0:7),13 .+ (0:14))
        Cbig = rand(ComplexF32,(40,40,40))
        C = view(Cbig,3 .+ 2*(0:8),13 .+ (0:8),7 .+ 3*(0:7))
        Acopy = tensorcopy(A,1:4)
        Bcopy = tensorcopy(B,1:3)
        Ccopy = tensorcopy(C,1:3)
        α = randn(Float64)
        β = randn(Float64)
        Ccopy = β*Ccopy
        for d = 1 .+ (0:8),a = 1 .+ (0:8),e = 1 .+ (0:7)
            for b = 1 .+ (0:14),c = 1 .+ (0:6)
                Ccopy[d,a,e] += α*A[a,b,c,d]*conj(B[c,e,b])
            end
        end
        TensorOperations.tensorcontract!(α,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'C',β,C,[:d,:a,:e])
        @test C ≈ Ccopy
        @test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(α,A,[:a,:b,:c,:a],'N',B,[:c,:e,:b],'N',β,C,[:d,:a,:e])
        @test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(α,A,[:a,:b,:c,:d],'N',B,[:c,:b],'N',β,C,[:d,:a,:e])
        @test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(α,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'N',β,C,[:d,:e])
        @test_throws DimensionMismatch TensorOperations.tensorcontract!(α,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'N',β,C,[:d,:e,:a])
    end
end
