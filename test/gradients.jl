
using Flux
using Flux.Tracker: TrackedArray, gradcheck, back!, data, grad

gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...) # from Flux tests
gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

@testset "gradients" begin
    @testset "flux gradients" begin

        ## basics
        a = rand(4)
        ab = rand(4,5)
        abc = rand(4,5,6)

        pa = param(a)
        pab = param(ab)
        pabc = param(abc)

        @tensor a2[i] := a[i] + a[i]
        @tensor pa2[i] := pa[i] + pa[i]

        @test pa2 isa TrackedArray
        @test data(pa2) == a2

        @tensor s[] := a[i] * a[i]
        @tensor ps[] := pa[i] * pa[i]

        @test ps isa TrackedArray
        @test data(pa2) == a2

        ## add!
        r32 = randn(3,2);
        add1(x) = @tensor S[i,j] := 2 * x[i,j] + 3 * r32[j,i]
        @test gradtest(add1, (2,3))

        add2(y) = @tensor S[i,j] := 2 * r32[i,j] + 3 * y[j,i]
        @test gradtest(add2, (2,3))

        r312 = randn(3,1,2);
        add3(x) = @tensor S[k,j,i] := 0.11 * x[i,j,k] - 33 * r312[k,i,j]
        @test gradtest(add3, (1,2,3))

        ## trace!
        tr1(x) = @tensor T[k] := 22 * x[i,i,k]
        @test gradtest(tr1, (3,3,4))

        tr2(x) = @tensor T[k] := 22 * x[i,i,k,j,j]
        @test gradtest(tr2, (3,3,4,7,7))

        tr3add(x) = prod(@tensor R[i,j] := 5 * x[k,i,k,j] + 7 * r32[j,i])
        @test gradcheck(tr3add, rand(7,2,7,3))

        ## contract! A
        con1(x) = @tensor C[i,j] := 5 * x[i,k] * r32[k,j]
        @test gradtest(con1, (2,3))

        r22 = rand(2,2);
        con2add(x) = @tensor C[i,j] := 5 * x[i,k] * r32[k,j] + 7 * r22[j,i]
        @test gradtest(con2add, (2,3))

        con3(x) = @tensor C[i,j,m,n] := x[i,j,k] * r312[k,m,n]
        @test gradtest(con3, (1,2,3))

        con4(x) = @tensor C[i,m] := x[i,kk,k] * r312[k,m,kk]
        @test gradtest(con4, (1,2,3))

        con5(x) = @tensor C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]
        @test con5(rand(1,2,3)) isa Array
        @test con5(rand(1,2,3) |> param) isa TrackedArray
        con5(rand(1,2,3) |> param)[1] |> back! 
        @test gradtest(con5, (1,2,3)) 

        r392 = randn(3,9,2);
        con6(x) = @tensor C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
        @test gradtest(con6, (9,2,3))

        con7(x) = @tensor C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
        @test gradtest(con7, (9,2,3))

        ## contract! B
        con8b(x) = @tensor K[i,j] := 5 * r32[i,k] * x[k,j]
        @test gradtest(con8b, (2,3))

        con9b(x) = @tensor K[i,j,m,n] := r312[i,j,k] * x[m,k,n]
        @test gradtest(con9b, (1,2,3))

        con10b(x) = @tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n]
        @test gradtest(con10b, (9,2,3))

        r3399 = randn(3,3,9,9);
        con11add(x) = @tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n] + 7 * r3399[n,i,j,m]
        @test gradtest(con11add, (9,2,3))

        con12add(x) = @tensor K[n,j,m,i] := r392[i,j,k] * x[m,n,k] + 7 * r3399[n,i,j,m] - r3399[i,n,m,j]
        @test gradtest(con12add, (9,3,2))

        con13(x) = @tensor K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]
        @test gradtest(con13, (3,3,9,9))

        r33 = rand(3,3);
        con14(x) = @tensor K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]
        @test gradtest(con14, (3,3,9,9))

        con15(x) = @tensor K[i,j] := r3399[a,a,j,k] * x[b,b,k,i] + 3.14 * r33[a,b] * x[a,c,k,i] * x[c,b,j,k]
        @test gradtest(con15, (3,3,9,9))

        con16(x) = @tensor K[i,j] := r3399[a,b,j,k] * x[b,a,k,i] + 3.14 * r33[a,b] * x[a,c,k,i] * x[c,b,j,k]
        @test con16(randn(3,3,9,9)) isa Array
        @test con16(randn(3,3,9,9) |> param) isa TrackedArray
        @test gradtest(con16, (3,3,9,9))

    end
    # @testset "zygote gradients?" begin

    #     r32 = randn(3,2);
    #     add2(y) = @tensor S[i,j] := 2 * r32[i,j] + 3 * y[j,i]
    #     con8b(x) = @tensor K[i,j] := 5 * r32[i,k] * x[k,j]

    #     x23 = rand(2,3);
    #     Flux.gradient(x -> sum(sin, add2(x)), x23)[1].data
    #     Zygote.gradient(x -> sum(sin, add2(x)), x23)[1] # huge error!

    # end
end
