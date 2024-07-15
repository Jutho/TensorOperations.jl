# Backends and Allocators

The `TensorOperations` package is designed to provide powerful tools for performing tensor computations efficiently.
In advanced use cases, it can be desirable to squeeze the last drops of performance out of the library, by experimenting with either different micro-optimized implementations of the same operation, or by altering the memory management system.
Here, we detail how to access these functionalities. Note that all of the backend and allocator
types documented below are not exported, so as not to pollute the name space and since they
will typically only be manually configured in expert use cases.

## Backends

### Backend Selection

`TensorOperations` supports multiple backends for tensor contractions, allowing users to choose different implementations based on their specific needs.
While special care is taken to ensure good defaults, we also provide the flexibility to select a backend manually.
This can be achieved in a variety of ways:

1. **Global setting**: The default backend can be set globally on a per-type basis, as well as a per-function basis. This is achieved by hooking into the implementation of the default backend selection procedure. In particular, this procedure ends up calling [`select_backend`](@ref)`, which can be overloaded to return a different backend.

2. **Local setting**: Alternatively, the backend can be set locally for a specific call to either [`@tensor`](@ref), [`ncon`](@ref) or the function-based interface. Both `@tensor` and `ncon` accept a keyword argument `backend`, which will locally override the default backend selection mechanism. The result is that the specified backend will be inserted as a final argument to all calls of the primitive tensor operations. This is also how this can be achieved in the function-based interface.

```julia
using TensorOperations
mybackend = TensorOperations.StridedNative()

# inserting a backend into the @tensor macro
@tensor backend = mybackend A[i,j] := B[i,k] * C[k,j]

# inserting a backend into the ncon function
D = ncon([A, B, C], [[1, 2], [2, 3], [3, 1]]; backend=mybackend)

# inserting a backend into the function-based interface
tensoradd(A, pA, conjA, B, pB, conjB, α, β, mybackend)
```

### Available Backends

`TensorOperations` provides some options for backends out-of-the box.
In particular, the following backends are available:

```@docs
TensorOperations.DefaultBackend
TensorOperations.BaseCopy
TensorOperations.BaseView
TensorOperations.StridedNative
TensorOperations.StridedBLAS
TensorOperations.cuTENSORBackend
```

Here, arrays that are strided are typically handled most efficiently by the `Strided.jl`-based backends.
By default, the `StridedBLAS` backend is used for element types that support BLAS operations, as it seems that the performance gains from using BLAS outweigh the overhead of sometimes having to allocate intermediate permuted arrays.

On the other hand, the `BaseCopy` and `BaseView` backends are used for arrays that are not strided.
These are designed to be as general as possible, and as a result are not as performant as specific implementations.
Nevertheless, they can be useful for debugging purposes or for working with custom tensor types that have limited support for methods outside of `Base`.

Finally, we also provide a `cuTENSORBackend` for use with the `cuTENSOR.jl` library, which is a NVidia GPU-accelerated tensor contraction library.
This backend is only available through a package extension for `cuTENSOR`.

### Custom Backends

Users can also define their own backends, to facilitate experimentation with new implementations.
This can be done by defining a new type that is a subtype of `AbstractBackend`, and dispatching on this type in the implementation of the primitive tensor operations.
In particular, the only required implemented methods are [`tensoradd!`](@ref), [`tensortrace!`](@ref), [`tensorcontract!`](@ref).

For example, [`TensorOperationsTBLIS`](https://github.com/lkdvos/TensorOperationsTBLIS.jl) is a wrapper that provides a backend for tensor contractions using the [TBLIS](https://github.com/devinamatthews/tblis) library.

## Allocators

Evaluating complex tensor networks is typically done most efficiently by pairwise operations.
As a result, this procedure often requires the allocation of many temporary arrays, which can affect performance for certain operations.
To mitigate this, `TensorOperations` exposes an allocator system, which allows users to more finely control the allocation of both output tensors and temporary tensors.

In particular, the allocator system is used in multiple ways:
As mentioned before, it can be used to allocate and free the intermediate tensors that are required to evaluate a tensor network in a pairwise fashion.
Additionally, it can also be used to allocate and free temporary objects that arise when reshaping and permuting input tensors, for example when making them compatible with BLAS instructions.

### Allocator Selection

The allocator system can only be accessed *locally*, by passing an allocator to the `@tensor` macro, the `ncon` function, or the function-based interface.

```julia
using TensorOperations
myallocator = TensorOperations.ManualAllocator()

# inserting a backend into the @tensor macro
@tensor allocator = myallocator A[i,j] := B[i,k] * C[k,j]

# inserting an allocator into the ncon function
D = ncon([A, B, C], [[1, 2], [2, 3], [3, 1]]; allocator=myallocator)

# inserting a backend into the function-based interface
tensoradd(A, pA, conjA, B, pB, conjB, α, β, DefaultBackend(), myallocator)
```

Important to note here is that the backend system is prioritized over the allocator system.
In particular, this means that the backend will be selected **first**, while only then the allocator should be inserted.

### Available Allocators

`TensorOperations` also provides some options for allocators out-of-the box.

```@docs
TensorOperations.DefaultAllocator
TensorOperations.ManualAllocator
```

By default, the `DefaultAllocator` is used, which uses Julia's built-in memory management system.
Optionally, it can be useful to use the `ManualAllocator`, as the manual memory management reduces the pressure on the garbage collector.
In particular in multi-threaded applications, this can sometimes lead to a significant performance improvement.

Finally, users can also opt to use the `Bumper.jl` system, which pre-allocates a slab of memory that can be re-used afterwards.
This is available through a package extension for `Bumper`.
Here, the `allocator` object is just the provided buffers, which are then used to store the intermediate tensors.

```julia
using TensorOperations, Bumper
buf = Bumper.default_buffer()
@no_escape buf begin
    @tensor allocator = buf A[i,j] := B[i,k] * C[k,j]
end
```
For convenience, the construction above is also provided in a specialized macro form which is fully equivalent:

```@docs
@butensor
```

When using the `CuTENSORBackend()` and no allocator is specified, it will automatically select the
allocator `CUDAAllocator()`, which will create new temporaries as `CuArray` objects. However,
`CUDAAllocator` has three type parameters which can be used to customize the behavior of the allocator
with respect to temporaries, as well as input and output tensors.

```@docs
TensorOperations.CUDAAllocator
```

### Custom Allocators

Users can also define their own allocators, to facilitate experimentation with new implementations.
Here, no restriction is made on the type of the allocator, and any object can be passed as an allocator.
The required implementated methods are [`tensoralloc`](@ref) and [`tensorfree`](@ref).

