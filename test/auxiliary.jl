@testset "methods for tensoropt" begin
    @testset "set methods" for T in (UInt32,UInt64,UInt128,BitSet,BitVector)
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

    @testset "poly" begin
        using TensorOperations: Power, degree, Poly
        x = Power{:x}(1,1)
        x⁰ = Power{:x}(1,0)
        x⁵= Power{:x}(1,5)
        @test x^5 == x⁵ == x*x*x*x*x
        @test x^0 == x⁰ == one(x)
        @test zero(x) == 0*x
        @test (5*x)/5 == x
        @test degree(x) == 1
        @test x + x^3 + x^5 isa Poly
        p = x - x^3 + x^5
        coeffs = [0,1,0,-1,0,1]
        @test p == Poly{:x}(coeffs)
        @test 2.5*p == Poly{:x}(2.5*coeffs) == p*2.5
        @test p/2 == Poly{:x}(coeffs/2) == 2\p
        @test p*p == x^10+x^6+x^2-2*x^8+2*x^6-2*x^4 == (x^10+x^6+x^2+1)-2*(x^8-x^6+x^4)-1
        p = x + x^3 + x^5
        for i = 0:6
            @test p[i] == isodd(i)
        end
        @test p + p/2 == 1.5*p
        @test p*one(p) == p == p*one(x)
        @test p*zero(p) == 0 == p*zero(x)
        @test p <= p*x
        @test x*p == p*x
    end

    @testset "cost methods" begin

    end
end

@testset "cache" begin
    c = TensorOperations.LRU{Int,Int}(; maxsize = 1024) # 1 kilobyte cache
    numints = div(1024,sizeof(Int))
    sizehint!(c, numints)
    for i = 1:10
        c[i] = i
    end
    @test haskey(c, 5)
    @test c[5] == 5
    c[5] = 6
    @test c[5] == 6
    @test !haskey(c, 11)
    @test length(c) == 10
    @test Set(collect(c)) == Set([i=>i for i = 1:10])
    @test !haskey(c, 11)
    @test get!(()->11, c, 10) == 10
    @test c[10] == 10
    @test get!(()->11, c, 11) == 11
    @test haskey(c, 11)
    @test c[11] == 11
    @test !haskey(c, 12)
    @test get!(c, 12, 10) == 10
    @test c[10] == 10
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
