@testset "@butensor dependency check" begin
    @test_throws ArgumentError begin
        ex = :(@butensor A[a, b, c, d] := B[a, b, c, d])
        macroexpand(Main, ex)
    end
end

using Bumper
@testset "Bumper tests with eltype $T" for T in (Float32, ComplexF64)
    D1, D2, D3 = 30, 40, 20
    d1, d2 = 2, 3

    A1 = randn(T, D1, d1, D2)
    A2 = randn(T, D2, d2, D3)
    ρₗ = randn(T, D1, D1)
    ρᵣ = randn(T, D3, D3)
    H = randn(T, d1, d2, d1, d2)

    @tensor begin
        HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                               ρᵣ[c', c] *
                               H[s1, s2, t1, t2]
    end

    # butensor implementation
    @butensor begin
        HRAA2[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                               ρᵣ[c', c] *
                               H[s1, s2, t1, t2]
    end
    @test HRAA2 isa Array{T,4}
    @test HRAA2 ≈ HRAA1

    # manual equivalent
    @no_escape @tensor allocator = default_buffer() begin
        HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                               ρᵣ[c', c] *
                               H[s1, s2, t1, t2]
    end
    @test HRAA3 isa Array{T,4}
    @test HRAA3 ≈ HRAA1

    # new buffer / completely manual
    slabsize = typeof(default_buffer()).parameters[1] >> 1 # take slab size half as big
    slabbuf = SlabBuffer{slabsize}()
    begin
        local cp = Bumper.checkpoint_save(slabbuf)
        @tensor allocator = slabbuf begin
            HRAA4[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                                   ρᵣ[c', c] *
                                   H[s1, s2, t1, t2]
        end
        bufferlength = slabsize * length(slabbuf.slabs)
        Bumper.checkpoint_restore!(cp)
    end
    @test HRAA4 isa Array{T,4}
    @test HRAA4 ≈ HRAA1

    # allocbuffer
    allocbuf = AllocBuffer(bufferlength)
    @no_escape allocbuf @tensor allocator = allocbuf begin
        HRAA5[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                               ρᵣ[c', c] *
                               H[s1, s2, t1, t2]
    end
    @test HRAA5 isa Array{T,4}
    @test HRAA5 ≈ HRAA1
end
