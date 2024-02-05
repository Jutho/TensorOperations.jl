using CairoMakie
using Statistics
using DataFrames
using TensorOperations
using TensorOperations: Backend

include("benchmark_utils.jl")

measurement = minimum
n_blas = 8
n_strided = 8
use_mkl = true


# Compute flops
num_els = compute_permute_size.(readlines(PERMUTATIONS_SPECS))
num_flops = compute_contract_ops.(readlines(CONTRACTIONS_SPECS))

fn = result_filename("HEAD", use_mkl, n_blas, n_strided)
res = PkgBenchmark.readresults(fn)

tensoradd_result = res.benchmarkgroup["tensoradd!"]
df = select(DataFrame(BenchmarkTools.leaves(tensoradd_result)),
            1 => (x -> eval.(Meta.parse.(getindex.(x, 1)))) => :T,
            1 => (x -> eval.(Meta.parse.(getindex.(x, 2)))) => :backend,
            1 => (x -> parse.(Int, getindex.(x, 3))) => :case,
            2 => (x -> getfield.(measurement.(x), :time)) => :time,
            2 => (x -> getfield.(measurement.(x), :memory)) => :memory)

backends = unique(df.backend)
eltypes = unique(df.T)

transform!(df,
           [:case, :time, :T] => ((i, t, T) -> getindex.(Ref(num_els), i) ./ t .* 1e9 ./ 2^30 .* sizeof(T)) => :flops,
           :backend => ByRow(x -> findfirst(isequal(x), backends)) => :backend_index)

colors = Makie.wong_colors()

backend_sym(::TensorOperations.Backend{S}) where S = string(S)

f_tensoradd = let
    f = Figure(; size=(800, 600), title="tensoradd!")
    
    sidetitle = Label(f[1:4, 0], "Bandwidth [GiB/s]", rotation=π/2)
    supertitle = Label(f[0, 0:2], "tensoradd!"; halign=:center, fontsize=20, font=:bold)
    
    ax_float32 = Axis(f[1, 1]; title="Float32", 
                      xticks=(Int[], String[]))
    df_float32 = filter(row -> row.T == Float32, df)
    barplot!(ax_float32, df_float32.case, df_float32.flops;
             dodge=df_float32.backend_index, color=colors[df_float32.backend_index])

    ax_float64 = Axis(f[2, 1]; title="Float64", 
                      xticks=(Int[], String[]))
    df_float64 = filter(row -> row.T == Float64, df)
    barplot!(ax_float64, df_float64.case, df_float64.flops;
             dodge=df_float64.backend_index, color=colors[df_float64.backend_index])
    
    ax_complexf32 = Axis(f[3, 1]; title="ComplexF32", 
                      xticks=(Int[], String[]))
    df_complexf32 = filter(row -> row.T == ComplexF32, df)
    barplot!(ax_complexf32, df_complexf32.case, df_complexf32.flops;
             dodge=df_complexf32.backend_index, color=colors[df_complexf32.backend_index])
    
    ax_complexf64 = Axis(f[4, 1]; title="ComplexF64", 
                      xticks=(Int[], String[]))
    df_complexf64 = filter(row -> row.T == ComplexF64, df)
    barplot!(ax_complexf64, df_complexf64.case, df_complexf64.flops;
             dodge=df_complexf64.backend_index, color=colors[df_complexf64.backend_index])
    
    # g_legend = f[1:2, 2] = GridLayout()
    Legend(f[1:4, 2], [PolyElement(; polycolor=colors[i]) for i in 1:length(backends)],
           backend_sym.(backends), "Backends")
    
    f
end


tensorcontract_result = res.benchmarkgroup["tensorcontract!"]
df = select(DataFrame(BenchmarkTools.leaves(tensorcontract_result)),
            1 => (x -> eval.(Meta.parse.(getindex.(x, 1)))) => :T,
            1 => (x -> eval.(Meta.parse.(getindex.(x, 2)))) => :backend,
            1 => (x -> parse.(Int, getindex.(x, 3))) => :case,
            2 => (x -> getfield.(measurement.(x), :time)) => :time,
            2 => (x -> getfield.(measurement.(x), :memory)) => :memory)

backends = unique(df.backend)
eltypes = unique(df.T)

transform!(df,
           [:case, :time, :T] => ((i, t, T) -> getindex.(Ref(num_flops), i) ./ t .* 1e9 ./ 2^30) => :flops,
           :backend => ByRow(x -> findfirst(isequal(x), backends)) => :backend_index)

f_tensoracontract = let
    f = Figure(; size=(800, 600), title="tensorcontract!")

    sidetitle = Label(f[1:4, 0], "GFLOPS"; rotation=π / 2)
    supertitle = Label(f[0, 0:2], "tensorcontract!"; halign=:center, fontsize=20, font=:bold)

    ax_float32 = Axis(f[1, 1]; title="Float32",
                      xticks=(Int[], String[]))
    df_float32 = filter(row -> row.T == Float32, df)
    barplot!(ax_float32, df_float32.case, df_float32.flops;
             dodge=df_float32.backend_index, color=colors[df_float32.backend_index])

    ax_float64 = Axis(f[2, 1]; title="Float64",
                      xticks=(Int[], String[]))
    df_float64 = filter(row -> row.T == Float64, df)
    barplot!(ax_float64, df_float64.case, df_float64.flops;
             dodge=df_float64.backend_index, color=colors[df_float64.backend_index])

    ax_complexf32 = Axis(f[3, 1]; title="ComplexF32",
                         xticks=(Int[], String[]))
    df_complexf32 = filter(row -> row.T == ComplexF32, df)
    barplot!(ax_complexf32, df_complexf32.case, df_complexf32.flops;
             dodge=df_complexf32.backend_index, color=colors[df_complexf32.backend_index])

    ax_complexf64 = Axis(f[4, 1]; title="ComplexF64",
                         xticks=(Int[], String[]))
    df_complexf64 = filter(row -> row.T == ComplexF64, df)
    barplot!(ax_complexf64, df_complexf64.case, df_complexf64.flops;
             dodge=df_complexf64.backend_index, color=colors[df_complexf64.backend_index])

    # g_legend = f[1:2, 2] = GridLayout()
    Legend(f[1:4, 2], [PolyElement(; polycolor=colors[i]) for i in 1:length(backends)],
           backend_sym.(backends), "Backends")

    f
end



# barplot!(ax, df.case, df.flops;
    # dodge=df.backend_index)


# Ts = keys(tensoradd_result)

# p_speed = plot(; title="tensoradd!", xticks=nothing, ylabel="Bandwidth (GiB/s)",
            #    legend=:topleft)