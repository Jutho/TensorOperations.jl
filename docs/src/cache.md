# Cache for temporaries

Contracting a sequence of tensors is provably most efficient (in terms of number of computations)
by contracting them pairwise. However, this requires that several intermediate results need
to be stored. In addition, if the contraction needs to be performed as a BLAS matrix multiplication
(which is typically the fastest choice), every tensor typically needs an additional permuted
copy that is compatible with the implementation of the contraction as multiplication. All these
temporary arrays, which can be large, put a a lot of pressure on Julia's garbage collector,
and the total time spent in the garbage collector can become significant.

That's why there is now a functionality to store intermediate results in a package wide cache,
where they can be reused upon a next run, either a next iteration if the tensor contraction
appears within the body of a loop, or on the next function call if it appears directly
within a given function. This mechanism only works with the `@tensor` macro, not with the
function-based interface.

The `@tensor` macro expands the given expression and immediately generates the code to create
the necessary temporaries. It associates with each of them a random symbol (`gensym()`) and
also uses this as an identifier in a package wide global cache variable, which is an `LRU{Symbol,Any}`
(least recently used cache) from the package [LRUCache.jl](https://github.com/JuliaCollections/LRUCache.jl).
In particular, the cache has a certain maximum length, and once more objects are added to it,
the least recently used objects will be purged. The default length in TensorOperations.jl is
currently 50, but can easily be changed using the objects below.

## Enabling and disabling the cache
The use of the cache can be enabled or disabled using
```@docs
enable_cache
disable_cache
clear_cache
cachesize
```

## Cache and multithreading
The `LRU` cache currently used is not thread safe, i.e. if there is any chance that different
threads will run the same `@tensor` expression block, you should `disable_cache()` or run the
risk of obtaining incorrect results.
