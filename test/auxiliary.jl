@testset "macro methods" begin
    @testset "tensorexpressions" begin
        using TensorOperations: isassignment, isdefinition, getlhs, getrhs, isindex,
            istensor, isgeneraltensor, istensorexpr, isscalarexpr, hastraceindices,
            hastraceindices, getindices, getallindices,
            normalizeindex, instantiate_scalar, instantiate_eltype,
            decomposetensor, decomposegeneraltensor

        @test isassignment(:(a[-1,-2,-3] = b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test isassignment(:(a[-1,-2,-3] += b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test isassignment(:(a[-1,-2,-3] -= b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isassignment(:(a[-1,-2,-3] := b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isassignment(:(a[-1,-2,-3] ≔ b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isassignment(:(b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))

        @test !isdefinition(:(a[-1,-2,-3] = b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isdefinition(:(a[-1,-2,-3] += b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isdefinition(:(a[-1,-2,-3] -= b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test isdefinition(:(a[-1,-2,-3] := b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test isdefinition(:(a[-1,-2,-3] ≔ b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))
        @test !isdefinition(:(b[-1,-2,1]*c[1,-3]+d[-2,-3,-1]))

        lhs = :(a[-1,-2,-3])
        rhs = :(b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])
        for (getside,side) in ((getlhs, lhs), (getrhs, rhs))
            @test getside(:(a[-1,-2,-3] = b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])) == side
            @test getside(:(a[-1,-2,-3] += b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])) == side
            @test getside(:(a[-1,-2,-3] -= b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])) == side
            @test getside(:(a[-1,-2,-3] := b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])) == side
            @test getside(:(a[-1,-2,-3] ≔ b[-1,-2,1]*c[1,-3]+d[-2,-3,-1])) == side
        end

        @test isindex(:a)
        @test normalizeindex(:a) == :a
        @test isindex(:(a'))
        @test normalizeindex(:(a')) == :(a′)
        @test isindex(:(a'''))
        @test normalizeindex(:(a''')) == :(a′′′)
        @test isindex(:(β))
        @test normalizeindex(:(β)) == :(β)
        @test isindex(:(β'))
        @test normalizeindex(:(β')) == :(β′)
        @test isindex(:(3))
        @test normalizeindex(:(3)) == 3
        @test isindex(:(-5))
        @test normalizeindex(:(-5)) == -5
        @test !isindex(:('a'))
        @test !isindex(:(a+b))
        @test !isindex(:("x"))
        @test !isindex(:(5.1))

        @test istensor(:(a[1,2,3]))
        @test decomposetensor(:(a[1,2,3])) == (:a, Any[1,2,3], Any[])
        @test istensor(:(a[5][a b c]))
        @test decomposetensor(:(a[5][a b c])) == (:(a[5]), Any[:a,:b,:c], Any[])
        @test istensor(:(cos(y)[a b c; 1 2 3]))
        @test decomposetensor(:(cos(y)[a b c; 1 2 3])) == (:(cos(y)), Any[:a,:b,:c], Any[1,2,3])
        @test istensor(:(x[(); (1,2,3)]))
        @test decomposetensor(:(x[(); (1,2,3)])) == (:x, Any[], Any[1,2,3])
        @test !istensor(:(2*a[1,2,3]))
        @test !istensor(:(a[1 2 3; 4 5 6; 7 8 9]))
        @test !istensor(:(conj(a[5][a b c])))
        @test !istensor(:(cos(y)[a b c; 1 2 3]*b[4,5]))
        @test !istensor(:(3+5))

        @test isgeneraltensor(:(conj(a[1,2,3])))
        @test decomposegeneraltensor(:(conj(a[1,2,3]))) == (:a, [1,2,3], [], instantiate_scalar(:(conj(1))), true)
        @test isgeneraltensor(:(x*a[5][a b c]))
        @test decomposegeneraltensor(:(x*a[5][a b c])) == (:(a[5]), [:a,:b,:c], [], instantiate_scalar(:(x*1)), false)
        @test isgeneraltensor(:(3*conj(a*cos(y)[a b c; 1 2 3])))
        @test decomposegeneraltensor(:(3*conj(a*cos(y)[a b c; 1 2 3]))) == (:(cos(y)),  Any[:a,:b,:c],  Any[1,2,3], instantiate_scalar(:(3*conj(a*1))), true)
        @test !isgeneraltensor(:(1/a[1,2,3]))
        @test !isgeneraltensor(:(a[1 2 3; 4 5 6]\x))
        @test !isgeneraltensor(:(cos(y)[a b c; 1 2 3]*b[4,5]))
        @test !isgeneraltensor(:(3+5))

        @test hastraceindices(:(a[x,y,z,1,x]))
        @test hastraceindices(:(a[x y z;1 x]))
        @test !hastraceindices(:(a[-1,y,z,1,x]))
        @test !hastraceindices(:(a[-1 y z;1 x]))

        @test isscalarexpr(:a)
        @test isscalarexpr(2.)
        @test isscalarexpr(:(2. + im*3))
        @test isscalarexpr(:(3*a+c))
        @test isscalarexpr(:(sin(x)+exp(-y)))
        @test isscalarexpr(:(scalar(a[x,y]*b[y,x])))
        @test !isscalarexpr(:(a[x,y]*b[y,x]))
        @test !isscalarexpr(:(3*scalar(a[x,y]*b[y,x]) + conj(c[z])))

        @test instantiate_eltype(:(a[1,2,3]*b[3,4,5]+c[1,2,4,5])) == :(promote_type(promote_type(eltype(a), eltype(b)), eltype(c)))
    end

    @testset "parsecost" begin
        using TensorOperations: parsecost, Power
        @test parsecost(:(3/5)) === 3/5
        @test parsecost(:(5+2)) === 5+2
        @test parsecost(:(Int128(2*8))) === Int128(2*8)
        @test parsecost(:(float(2*8))) === float(2*8)
        @test parsecost(:(big(2*8))) isa BigInt
        @test parsecost(:(x)) == Power{:x}(1,1)
        @test parsecost(:(3*x^3)) == Power{:x}(3,3)
        @test parsecost(:(3*x^3+2)) == Power{:x}(3,3)+2
        @test parsecost(:((3*x^3-2)/5)) == (Power{:x}(3,3)-2)/5
    end
end

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
