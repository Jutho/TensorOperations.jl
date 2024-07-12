module TensorOperationsBumperExt

using TensorOperations
using Bumper

function TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp},
                                      buf::Union{SlabBuffer,AllocBuffer}) where {A<:AbstractArray,
                                                                                 istemp}
    # TODO: remove the `ndims` check if this is fixed in Bumper / StrideArraysCore
    if istemp & ndims(A) > 0
        return Bumper.alloc!(buf, eltype(A), structure...)
    else
        return TensorOperations.tensoralloc(A, structure, Val(istemp))
    end
end

function TensorOperations.blas_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
                                         allocator::Union{SlabBuffer,AllocBuffer})
    @no_escape allocator begin
        C = Base.@invoke TensorOperations.blas_contract!(C::Any, A::Any, pA::Any,
                                                         conjA::Any,
                                                         B::Any, pB::Any,
                                                         conjB::Any, pAB::Any, α::Any,
                                                         β::Any,
                                                         allocator::Any)
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

end
