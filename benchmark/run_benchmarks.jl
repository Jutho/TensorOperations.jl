#!/usr/bin/env -S julia --color=yes --startup-file=no -t auto -O3 --

import Pkg; Pkg.activate(@__DIR__)

# parse input arguments
version_id = findfirst(isequal("-v"), ARGS)
version = isnothing(version_id) ? "HEAD" : ARGS[version_id + 1]

mkl_id = findfirst(isequal("-mkl"), ARGS)
use_mkl = isnothing(mkl_id) ? true : parse(Bool, ARGS[mkl_id + 1])

blas_id = findfirst(isequal("-blas"), ARGS)
blas_threads = isnothing(blas_id) ? 1 : parse(Int, ARGS[blas_id + 1])

strided_id = findfirst(isequal("-strided"), ARGS)
strided_threads = isnothing(strided_id) ? 1 : parse(Int, ARGS[strided_id + 1])

# setup environment
if use_mkl
    using MKL
end
using Strided: Strided
using LinearAlgebra: BLAS
import TensorOperations

using PkgBenchmark

# set threads
BLAS.set_num_threads(blas_threads)
Strided.set_num_threads(strided_threads)

# run benchmarks

include("benchmark_utils.jl")
blas_vendor = use_mkl ? "MKL" : "OpenBLAS"
resultfile = result_filename(version, use_mkl, blas_threads, strided_threads)

cfg = BenchmarkConfig(; id=nothing, env=Dict("MKL" => use_mkl, "BLAS_THREADS" => blas_threads, "STRIDED_THREADS" => strided_threads), juliacmd = `julia -t auto -O3`)

@info "BLAS vendor: $blas_vendor with $blas_threads threads"
@info "Strided threads: $strided_threads"
res = benchmarkpkg("TensorOperations", cfg; verbose=true, resultfile)
