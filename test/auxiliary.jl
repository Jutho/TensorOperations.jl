@testset "set methods for tensoropt" begin
    for T in (UInt32,UInt64,UInt128,BitSet,BitVector)
        maxint = 32
        A = Set(rand(1:maxint, 20))
        B = Set(rand(1:maxint, 20))
        U = union(A,B)
        I = intersect(A,B)
        S = setdiff(A,B)

        A2 = TensorOperations.storeset(T, A, maxint)
        B2 = TensorOperations.storeset(T, B, maxint)
        U2 = TensorOperations.storeset(T, U, maxint)
        I2 = TensorOperations.storeset(T, I, maxint)
        S2 = TensorOperations.storeset(T, S, maxint)

        @test TensorOperations._union(A2, B2) == U2
        @test TensorOperations._intersect(A2, B2) == I2
        @test TensorOperations._setdiff(A2, B2) == S2
        @test TensorOperations._isemptyset(TensorOperations.storeset(T, [], maxint))
        @test !TensorOperations._isemptyset(TensorOperations.storeset(T, [1], maxint))
    end
end

@testset "cache" begin
    c = TensorOperations.LRU{Int,Int}(; maxsize = 1024) # 1 kilobyte cache
    numints = div(1024,sizeof(Int))
    for i = 1:10
        c[i] = i
    end
    @test haskey(c, 5)
    @test !haskey(c, 11)
    @test c[5] == 5
    @test length(c) == 10
    @test Set(collect(c)) == Set([i=>i for i = 1:10])
    @test !haskey(c, 11)
    @test get!(()->11, c, 11) == 11
    @test haskey(c, 11)
    @test c[11] == 11
    @test !haskey(c, 12)
    @test get!(c, 12, 12) == 12
    @test haskey(c, 12)
    @test c[12] == 12
    delete!(c, 12)
    @test !haskey(c, 12)
    @test length(c) == 11
    @test !isempty(c)
    empty!(c)
    @test isempty(c)
    @test length(c) == 0
    for i = 1:(numints+1)
        c[i] = i
    end
    @test !haskey(c, 1)
    @test haskey(c, 2)
    TensorOperations.setsize!(c; maxsize = 2048)
    for i = 1:(numints+1)
        c[i] = i
    end
    @test haskey(c, 1)
end
