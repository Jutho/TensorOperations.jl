
# Implementation

## Building blocks

Under the hood, the implementation is  centered around the primitive operations: addition,
tracing and contraction. These operations are implemented for arbitrary strided arrays from
Julia Base, i.e. `Array`s, views with ranges thereof, and certain reshape operations. This
includes certain arrays that can only be determined to be strided on runtime, and does
therefore not coincide with the type union `StridedArray` from Julia Base. In fact, the
methods accept `AbstractArray` objects, but convert these to `(Unsafe)StridedView` objects
from the package [Strided.jl](https://github.com/Jutho/Strided.jl), and we refer to this
package for a more detailed discussion of which arrays are supported and why.

Nonetheless, the implementation can easily be extended to user defined types, especially if
they just wrap multidimensional data with a strided memory storage. The building blocks
resemble the functions discussed above, but have a different interface and are more
general. They are used both by the functions as well as by the `@tensor` macro, as
discussed below. Note that these functions are not exported.

The primitive tensor operations are captured by the following mutating methods
```@docs
add!
trace!
contract!
```
These are the central objects that should be overloaded by custom tensor types that would
like to be used within the `@tensor` environment.

Furthermore, it is essential to be able to construct new tensor objects that are similar
to existing ones, i.e. to place the result of the computation in case no output is
specified. In order to reuse temporary objects stored in the global cache, this method also
receives a candidate similar object, which it can return if it matches the requirements.
```@docs
checked_similar_from_indices
```
Note that the type of the cached object is not known to the compiler, as the cache stores
objects as `Any`. Therefore, the function `checked_similar_from_indices` should try to
restore the type information. By passing any object retrieved from the cache through this
function, type stability within the `@tensor` macro can then still be guaranteed.

Finally, there is a simple helper function

*   `add!(α, A, conjA, β, C, indCinA)`

    Implements `C = β*C+α*permute(op(A))` where `A` is permuted according to `indCinA` and
    `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable
    collection `indCinA` contains as nth entry the dimension of `A` associated with the nth
    dimension of `C`.

- `trace!(α, A, conjA, β, C, indCinA, cindA1, cindA2)`

  Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced, according to `indCinA`, `cindA1` and `cindA2`, and `op` is `conj` if `conjA=Val{:C}` or the identity map if `conjA=Val{:N}`. The indexable collection `indCinA` contains as nth entry the dimension of `A` associated with the nth dimension of `C`. The partial trace is performed by contracting dimension `cindA1[i]` of `A` with dimension `cindA2[i]` of `A` for all `i in 1:length(cindA1)`.

- `contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indCinoAB, [method])`

  Implements `C = β*C+α*contract(op(A),op(B))` where `A` and `B` are contracted according to `oindA`, `cindA`, `oindB`, `cindB` and `indCinoAB`. The operation `op` acts as `conj` if `conjA` or `conjB` equal `Val{:C}` or as the identity map if `conjA` (`conjB`) equal `Val{:N}`. The dimension `cindA[i]` of `A` is contracted with dimension `cindB[i]` of `B`. The `n`th dimension of C is associated with an uncontracted (open) dimension of `A` or `B` according to `indCinoAB[n] < NoA ? oindA[indCinoAB[n]] : oindB[indCinoAB[n]-NoA]` with `NoA=length(oindA)` the number of open dimensions of `A`.

  The optional argument `method` specifies whether the contraction is performed using BLAS matrix multiplication by specifying `Val{:BLAS}`, or using a native algorithm by specifying `Val{:native}`. The native algorithm does not copy the data but is typically slower. The BLAS-based algorithm is chosen by default, if the element type of the output array is in `Base.LinAlg.BlasFloat`.

## Index notation and the `@tensor` macro
