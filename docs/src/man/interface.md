# Interface

The `@tensor` macro rewrites tensor operations in terms of basic building blocks, such that
any tensor type that implements the following interface can be supported. In these methods,
`C` will indicate an output tensor which is changed in-place, while `A` and `B` denote input
tensors that are left unaltered. `pC`, `pA` and `pB` denote an `Index2Tuple`, a tuple of two
tuples that represents a permutation and partition of the original tensor indices. Finally,
`conjA` and `conjB` are symbols that are used to indicate if the input tensor should be
conjugated (`:C`) or used as-is (`:N`).

## Operations

```@docs
tensoradd!(::Any, ::Index2Tuple, ::Any, ::Symbol, ::Number, ::Number)
tensortrace!(::Any, ::Index2Tuple, ::Any, ::Index2Tuple, ::Symbol, ::Number, ::Number)
tensorcontract!(::Any, ::Index2Tuple, ::Any, ::Index2Tuple, ::Symbol, ::Any, ::Index2Tuple, ::Symbol, ::Number, ::Number)
tensorscalar
```

## Allocations

For some networks, it will be necessary to allocate output and/or intermediate tensors.
This process is split into the following hierarchy of functions, where custom tensor types can opt in at different levels.

By default, the process is split into three steps.

First, the scalar type `TC` of the resulting tensor is determined.
This is done by leveraging [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)'s `scalartype`, and promoting the results along with the types of any scalars that are present.

```@docs
TensorOperations.promote_add
TensorOperations.promote_contract
```

Then, the type and structure of the resulting tensor is determined.
The former represents all the information that is contained within the type, while the latter adds the required runtime information (e.g. array sizes, ...).

```@docs
TensorOperations.tensoradd_type
TensorOperations.tensoradd_structure
TensorOperations.tensorcontract_type
TensorOperations.tensorcontract_structure
```

Finally, the tensor is allocated, where a flag indicates if this is a temporary object, or one that will persist outside of the scope of the macro.
If the resulting tensor is a temporary object and its memory will not be freed by Julia's garbage collector, it can be explicitly freed by implementing `tensorfree!`, which by default does nothing.

```@docs
tensoralloc
tensorfree!
```

The `@tensor` macro will however only insert the calls to the following functions, which have a default implementation in terms of the functions above.

```@docs
TensorOperations.tensoralloc_add
TensorOperations.tensoralloc_contract
```

## Utility

Some of the optional keywords for `@tensor` can be accessed only after implementing the following utility functions:

```@docs
tensorcost
checkcontractible
```
