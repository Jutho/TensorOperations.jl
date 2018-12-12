
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

## Index notation and the `@tensor` macro
