using Deprecations
deps = map(x->x(), keys(Deprecations.all_deprecations))
# time: 2017-08-17 13:43:03 BST
# mode: julia
# time: 2017-08-17 13:43:08 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/TensorOperations.jl", deps)
# time: 2017-08-17 13:43:49 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/implementation/stridedarray.jl", deps)
# time: 2017-08-17 13:44:02 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/implementation/strides.jl", deps)
# time: 2017-08-17 13:44:10 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/implementation/recursive.jl", deps)
# time: 2017-08-17 13:44:16 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/implementation/kernels.jl", deps)
# time: 2017-08-17 13:44:26 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/implementation/indices.jl", deps)
# time: 2017-08-17 13:44:38 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/axpby.jl", deps)
# time: 2017-08-17 13:44:44 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/error.jl", deps)
# time: 2017-08-17 13:44:49 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/meta.jl", deps)
# time: 2017-08-17 13:45:06 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/stridedarray.jl"
	, deps)
# time: 2017-08-17 13:45:12 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/strideddata.jl"
	, deps)
# time: 2017-08-17 13:45:19 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/auxiliary/unique2.jl"
	, deps)
# time: 2017-08-17 13:45:33 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/functions/inplace.jl"
	, deps)
# time: 2017-08-17 13:45:38 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/functions/simple.jl"
	, deps)
# time: 2017-08-17 13:45:54 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/indexnotation/indexedobject.jl"
	, deps)
# time: 2017-08-17 13:46:05 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/indexnotation/product.jl"
	, deps)
# time: 2017-08-17 13:46:11 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/indexnotation/sum.jl"
	, deps)
# time: 2017-08-17 13:46:20 BST
# mode: julia
	Deprecations.edit_file("/home/data/.julia/v0.6/TensorOperations/src/indexnotation/tensormacro.jl"
	, deps)
# time: 2017-08-17 13:47:42 BST
# mode: julia
	Pkg.test("TensorOperations")
