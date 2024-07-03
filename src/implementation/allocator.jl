# ------------------------------------------------------------------------------------------
# DefaultAllocator
# ------------------------------------------------------------------------------------------
"""
    DefaultAllocator()

Default allocator for tensor operations if no explicit allocator is specified. This will
just use the standard constructor for the tensor type, and thus probably uses Julia's
default memory manager.
"""
struct DefaultAllocator end

"""
    CUDAAllocator{Mout,Min,Mtemp}()

Allocator that uses the CUDA memory manager and will thus allocate `CuArray` instances. The
parameters `Min`, `Mout`, `Mtemp`` can be any of the CUDA.jl memory types, i.e. 
`CUDA.DeviceMemory`, `CUDA.UnifiedMemory` or `CUDA.HostMemory`.
* `Mout` is used to determine how to deal with output tensors; with `Mout=CUDA.HostMemory`
  or `Mout=CUDA.UnifiedMemory` the CUDA runtime will ensure that the data is also available
  at in the host memory, and can thus be converted back to normal arrays using
  `unsafe_wrap(Array, outputtensor)`. If `Mout=CUDA.DeviceMemory` the data will remain on
  the GPU, untill an explict `Array(outputtensor)` is called.
* `Min` is used to determine how to deal with input tensors; with `Min=CUDA.HostMemory` the
  CUDA runtime will itself take care of transferring the data to the GPU, otherwise it is
  copied explicitly.
* `Mtemp` is used to allocate space for temporary tensors; it defaults to
  `CUDA.default_memory` which is `CUDA.DeviceMemory`. Only if many or huge temporary tensors
  are expected could it be useful to choose `CUDA.UnifiedMemory`.
"""
struct CUDAAllocator{M} end

# ------------------------------------------------------------------------------------------
# Generic implementation
# ------------------------------------------------------------------------------------------
tensorop(args...) = +(*(args...), *(args...))
"""
    promote_contract(args...)

Promote the scalar types of a tensor contraction to a common type.
"""
promote_contract(args...) = Base.promote_op(tensorop, args...)

"""
    promote_add(args...)

Promote the scalar types of a tensor addition to a common type.
"""
promote_add(args...) = Base.promote_op(+, args...)

"""
    tensoralloc_add(TC, A, pA, conjA, [istemp=false, allocator])

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensoradd!(C, A, pA, conjA)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `allocator` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_add(TC, A, pA::Index2Tuple, conjA::Bool, istemp::Bool=false,
                         allocator=DefaultAllocator())
    ttype = tensoradd_type(TC, A, pA, conjA)
    structure = tensoradd_structure(A, pA, conjA)
    return tensoralloc(ttype, structure, istemp, allocator)::ttype
end

"""
    tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB, [istemp=false, allocator])

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `allocator` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_contract(TC,
                              A, pA::Index2Tuple, conjA::Bool,
                              B, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple, istemp::Bool=false,
                              allocator=DefaultAllocator())
    ttype = tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# ------------------------------------------------------------------------------------------
# AbstractArray implementation
# ------------------------------------------------------------------------------------------

tensorstructure(A::AbstractArray) = size(A)
tensorstructure(A::AbstractArray, iA::Int, conjA::Bool) = size(A, iA)

function tensoradd_type(TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return Array{TC,sum(length.(pA))}
end

function tensoradd_structure(A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return size.(Ref(A), linearize(pA))
end

function tensorcontract_type(TC, A::AbstractArray, pA, conjA,
                             B::AbstractArray, pB, conjB, pAB, backend...)
    return Array{TC,sum(length.(pAB))}
end

function tensorcontract_structure(A::AbstractArray, pA, conjA,
                                  B::AbstractArray, pB, conjB, pAB, backend...)
    return let lA = length(pA[1])
        map(n -> n <= lA ? size(A, pA[1][n]) : size(B, pB[2][n - lA]), linearize(pAB))
    end
end

function tensoralloc(ttype, structure, istemp=false, allocator=DefaultAllocator())
    C = ttype(undef, structure)
    # fix an issue with undefined references for strided arrays
    if !isbitstype(scalartype(ttype))
        C = zerovector!!(C)
    end
    return C
end

function tensorfree!(C, allocator=DefaultAllocator())
    return nothing
end
