module TensorOperationsBumperExt

using TensorOperations
using TensorOperations: tensoralloc_add, tensoralloc_contract
using VectorInterface: One, Zero
using PrecompileTools
using Bumper
using Bumper: UnsafeArray

# Hack to normalize StridedView type to avoid too many specializations
# This is allowed because bumper ensures that the pointer won't be GC'd
# and we never return `parent(SV)` anyways.
@static if isdefined(Core, :Memory)
    function TensorOperations.wrap_stridedview(A::Bumper.UnsafeArray)
        mem_A = Base.unsafe_wrap(Memory{eltype(A)}, pointer(A), length(A))
        return TensorOperations.StridedView(mem_A, size(A), strides(A), 0, identity)
    end
end

function TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp},
                                      buf::Union{SlabBuffer,AllocBuffer}) where {A<:AbstractArray,
                                                                                 istemp}
    # TODO: remove the `ndims` check if this is fixed in Bumper / StrideArraysCore
    if istemp && ndims(A) > 0
        return Bumper.alloc!(buf, eltype(A), structure...)
    else
        return TensorOperations.tensoralloc(A, structure, Val(istemp))
    end
end

function TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β,
                                         backend, allocator::Union{SlabBuffer,AllocBuffer})
    @no_escape allocator begin
        C = Base.@invoke TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β,
                                                         backend, allocator::Any)
    end
    return C
end

function TensorOperations._butensor(src, ex...)
    buf_sym = gensym("buffer")
    cp_sym = gensym("checkpoint")
    res_sym = gensym("result")

    # TODO: there is no check for doubled tensor kwargs
    newex = quote
        $buf_sym = $(Expr(:call, GlobalRef(Bumper, :default_buffer)))
        $cp_sym = $(Expr(:call, GlobalRef(Bumper, :checkpoint_save), buf_sym))
        $res_sym = $(Expr(:macrocall, GlobalRef(TensorOperations, Symbol("@tensor")),
                          src, :(allocator = $buf_sym), ex...))
        $(Expr(:call, GlobalRef(Bumper, :checkpoint_restore!), cp_sym))
        $res_sym
    end
    return return Base.remove_linenums!(newex)
end

if PrecompileTools.workload_enabled(@__MODULE__)
    buf = typeof(Bumper.default_buffer())
    backend = TensorOperations.DefaultBackend

    # tensoradd!
    # ----------
    for T in TensorOperations.PRECOMPILE_ELTYPES
        for N in 0:(TensorOperations.PRECOMPILE_ADD_NDIMS)
            TA = Array{T,N}
            pA = Index2Tuple{N,0}
            TA_buf = UnsafeArray{T,N}
            for (C, A) in Iterators.product((TA, TA_buf), (TA, TA_buf))
                C == A == TA && continue
                precompile(tensoradd!, (C, A, pA, Bool, One, Zero))
                precompile(tensoradd!, (C, A, pA, Bool, T, Zero))
                precompile(tensoradd!, (C, A, pA, Bool, T, T))
            end

            for (A, istemp) in Iterators.product((TA, TA_buf), (Val{true}, Val{false}))
                precompile(tensoralloc_add, (T, A, pA, Bool, istemp, buf))
            end
        end
    end

    # tensortrace!
    # ------------
    for T in TensorOperations.PRECOMPILE_ELTYPES
        for N1 in 0:TensorOperations.PRECOMPILE_TRACE_NDIMS[1],
            N2 in 0:TensorOperations.PRECOMPILE_TRACE_NDIMS[2]

            TC = Array{T,N1}
            TA = Array{T,N1 + 2N2}
            p = Index2Tuple{N1,0}
            q = Index2Tuple{N2,N2}
            r = Index2Tuple{N1 + 2N2,0}

            TA_buf = UnsafeArray{T,N1 + 2N2}
            TC_buf = UnsafeArray{T,N1}

            for (C, A) in Iterators.product((TC, TC_buf), (TA, TA_buf))
                C == TC && A == TA && continue
                precompile(tensortrace!, (C, A, p, q, Bool, One, Zero))
                precompile(tensortrace!, (C, A, p, q, Bool, T, Zero))
                precompile(tensortrace!, (C, A, p, q, Bool, T, T))
            end

            # allocation re-uses tensoralloc_add
        end
    end

    # tensorcontract!
    # ---------------
    for T in TensorOperations.PRECOMPILE_ELTYPES
        for N1 in 0:TensorOperations.PRECOMPILE_CONTRACT_NDIMS[1],
            N2 in 0:TensorOperations.PRECOMPILE_CONTRACT_NDIMS[2],
            N3 in 0:TensorOperations.PRECOMPILE_CONTRACT_NDIMS[1]

            NA = N1 + N2
            NB = N2 + N3
            NC = N1 + N3
            TC, TA, TB = Array{T,NC}, Array{T,NA}, Array{T,NB}
            pA = Index2Tuple{N1,N2}
            pB = Index2Tuple{N2,N3}
            pAB = Index2Tuple{NC,0}

            TC_buf = UnsafeArray{T,NC}
            TA_buf = UnsafeArray{T,NA}
            TB_buf = UnsafeArray{T,NB}
            for (C, A, B) in Iterators.product((TC, TC_buf), (TA, TA_buf), (TB, TB_buf))
                precompile(tensorcontract!,
                           (C, A, pA, Bool, B, pB, Bool, pAB, One, Zero, backend,
                            buf))
                precompile(tensorcontract!,
                           (C, A, pA, Bool, B, pB, Bool, pAB, T, Zero, backend,
                            buf))
                precompile(tensorcontract!,
                           (C, A, pA, Bool, B, pB, Bool, pAB, T, T, backend, buf))
            end

            for (A, B, istemp) in
                Iterators.product((TA, TA_buf), (TB, TB_buf), (Val{true}, Val{false}))
                precompile(tensoralloc_contract,
                           (T, A, pA, Bool, B, pB, Bool, pAB, istemp, buf))
            end
        end
    end
end

end
