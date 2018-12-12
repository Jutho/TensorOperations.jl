# Cache for temporaries

Contracting a sequence of tensors is provably most efficient (in terms of number of
computations) by contracting them pairwise. However, this requires that several
intermediate results need to be stored. In addition, if the contraction needs to be
performed as a BLAS matrix multiplication (which is typically the fastest choice), every
tensor typically needs an additional permuted copy that is compatible with the
implementation of the contraction as multiplication. All these temporary arrays, which can
be large, put a a lot of pressure on Julia's garbage collector, and the total time spent in
the garbage collector can become significant.

That's why there is now a functionality to store intermediate results in a package wide
cache, where they can be reused upon a next run, either a next iteration if the tensor
contraction appears within the body of a loop, or on the next function call if it appears
directly within a given function. This mechanism only works with the `@tensor` macro, not
with the function-based interface.

The `@tensor` macro expands the given expression and immediately generates the code to
create the necessary temporaries. It associates with each of them a random symbol
(`gensym()`) and uses this as an identifier in a package wide global cache structure
`TensorOperations.cache`, the implementation of which is a least-recently used cache
dictionary borrowed from the package [LRUCache.jl](https://github.com/JuliaCollections/
LRUCache.jl), but adapted in such a way that it uses a maximum memory size rather than a
maximal number of objects. Thereto, it estimates the size of each object added to the cache
using `Base.summarysize` and, when discards objects once a certain memory limit is reached.

## Enabling and disabling the cache
The use of the cache can be enabled or disabled using
```@docs
enable_cache
disable_cache
```

Furthermore, the current total size of all the objects stored in the cache can be obtained
using the method `cachesize`, and `clear_cache` can be triggered to release all the objects
currently stored in the cache, such that they can be removed by Julia's garbage collector.
```@docs
cachesize
clear_cache
```

## Cache and multithreading
The `LRU` cache currently used is not thread safe, i.e. if there is any chance that
different threads will run tensor expressions using the `@tensor` environment, you should
`disable_cache()`.
