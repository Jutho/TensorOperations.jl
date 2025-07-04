# Precompilation

TensorOperations.jl has some support for precompiling commonly called functions.
The guiding philosophy is that often, tensor contractions are (part of) the bottlenecks of typical workflows,
and as such we want to maximize performance. As a result, we are choosing to specialize many functions which
may lead to a rather large time-to-first-execution (TTFX). In order to mitigate this, some of that work can
be moved to precompile-time, avoiding the need to re-compile these specializations for every fresh Julia session.

Nevertheless, TensorOperations is designed to work with a large variety of input types, and simply enumerating
all of these tends to lead to prohibitively large precompilation times, as well as large system images.
Therefore, there is some customization possible to tweak the desired level of precompilation, trading in
faster precompile times for fast TTFX for a wider range of inputs.

!!! compat "TensorOperations v5.2.0"

    Precompilation support requires at least TensorOperations v5.2.0.

## Defaults

By default, precompilation is disabled, but can be enabled for "tensors" of type `Array{T,N}`, where `T` and `N` range over the following values:

* `T` is either `Float64` or `ComplexF64`
* `tensoradd!` is precompiled up to `N = 5`
* `tensortrace!` is precompiled up to `4` free output indices and `2` pairs of traced indices
* `tensorcontract!` is precompiled up to `3` free output indices on both inputs, and `2` contracted indices

To enable precompilation with these default settings, you can *locally* change the `"precompile_workload"` key in the preferences.

```julia
using TensorOperations, Preferences
set_preferences!(TensorOperations, "precompile_workload" => true; force=true)
```

## Custom settings

The default precompilation settings can be tweaked to allow for more or less expansive coverage.
This is achieved through a combination of `PrecompileTools`- and `Preferences`-based functionality.

```julia
using TensorOperations, Preferences
set_preferences!(TensorOperations, "setting" => value; force=true)
```

Here **setting** and **value** can take on the following:

* `"precomple_eltypes"`: a `Vector{String}` that evaluate to the desired values of `T<:Number`
* `"precompile_add_ndims"`: an `Int` to specify the maximum `N` for `tensoradd!`
* `"precompile_trace_ndims"`: a `Vector{Int}` of length 2 to specify the maximal number of free and traced indices for `tensortrace!`.
* `"precompile_contract_ndims"`: a `Vector{Int}` of length 2 to specify the maximal number of free and contracted indices for `tensorcontract!`.

!!! note "Backends"

    Currently, there is no support for precompiling methods that do not use the default backend. If this is a
    feature you would find useful, feel free to contact us or open an issue.
