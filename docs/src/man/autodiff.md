# Automatic differentiation

TensorOperations offers experimental support for reverse-mode automatic diffentiation (AD)
through the use of [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). As the basic
operations are multi-linear, the vector-Jacobian products thereof can all be expressed in
terms of the operations defined in VectorInterface and TensorOperations. Thus, any custom
type whose tangent type also support these interfaces will automatically inherit
reverse-mode AD support.

As the [`@tensor`](@ref) macro rewrites everything in terms of the basic tensor operations,
the reverse-mode rules for these methods are supplied. However, because most AD-engines do
not support in-place mutation, effectively these operations will be replaced with a
non-mutating version. This is similar to the behaviour found in
[BangBang.jl](https://github.com/JuliaFolds/BangBang.jl), as the operations will be
in-place, except for the pieces of code that are being differentiated. In effect, this
amounts to replacing all assignments (`=`) with definitions (`:=`) within the context of
[`@tensor`](@ref).

!!! warning "Experimental"

    While some rudimentary tests are run, this feature is currently not incredibly well-tested.
    Because of the way it is implemented, the use of AD will tacitly replace mutating operations
    with a non-mutating variant. This might lead to unwanted bugs that are hard to track down.
    Additionally, for mixed scalar types their also might be unexpected or unwanted behaviour.