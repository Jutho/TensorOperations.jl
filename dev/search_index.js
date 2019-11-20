var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "TensorOperations.jl",
    "title": "TensorOperations.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#TensorOperations.jl-1",
    "page": "TensorOperations.jl",
    "title": "TensorOperations.jl",
    "category": "section",
    "text": "Fast tensor operations using a convenient Einstein index notation."
},

{
    "location": "#Table-of-contents-1",
    "page": "TensorOperations.jl",
    "title": "Table of contents",
    "category": "section",
    "text": "Pages = [\"index.md\", \"indexnotation.md\", \"functions.md\", \"cache.md\", \"implementation.md\"]\nDepth = 4"
},

{
    "location": "#Installation-1",
    "page": "TensorOperations.jl",
    "title": "Installation",
    "category": "section",
    "text": "Install with the package manager, pkg> add TensorOperations."
},

{
    "location": "#Package-features-1",
    "page": "TensorOperations.jl",
    "title": "Package features",
    "category": "section",
    "text": "A macro @tensor for conveniently specifying tensor contractions and index permutations   via Einstein\'s index notation convention. The index notation is analyzed at compile time.\nAbility to   optimize pairwise contraction order   using the @tensoropt macro. This optimization is performed at compile time, and the resulting contraction order is hard coded into the resulting expression. The similar macro @tensoropt_verbose provides more information on the optimization process.\nNew: a function ncon (for network contractor) for contracting a group of   tensors (a.k.a. a tensor network), as well as a corresponding @ncon macro that   simplifies and optimizes this slightly. Unlike the previous macros, ncon and @ncon   do not analyze the contractions at compile time, thus allowing them to deal with   dynamic networks or index specifications.\nSupport for any Julia Base array which qualifies as strided, i.e. such that its entries   are layed out according to a regular pattern in memory. The only exception are   ReinterpretedArray objects (implementation provided by Strided.jl, see below).   Additionally, Diagonal objects whose underlying diagonal data is stored as a strided   vector are supported. This facilitates tensor contractions where one of the operands is   e.g. a diagonal matrix of singular values or eigenvalues, which are returned as a   Vector by Julia\'s eigen or svd method.\nNew: Support for CuArray objects if used together with CuArrays.jl, by relying   on (and thus providing a high level interface into) NVidia\'s   cuTENSOR library.\nImplementation can easily be extended to other types, by overloading a small set of   methods.\nEfficient implementation of a number of basic tensor operations (see below), by relying   on Strided.jl and gemm from BLAS for   contractions. The latter is optional but on by default, it can be controlled by a   package wide setting via enable_blas() and disable_blas(). If BLAS is disabled or   cannot be applied (e.g. non-matching or non-standard numerical types), Strided.jl is   also used for the contraction.\nA package wide cache for storing temporary arrays that are generated when evaluating   complex tensor expressions within the @tensor macro (based on the implementation of   LRUCache). By default, the cache is   allowed to use up to the minimum of either 1GB or 25% of the total memory."
},

{
    "location": "#Tensor-operations-1",
    "page": "TensorOperations.jl",
    "title": "Tensor operations",
    "category": "section",
    "text": "TensorOperations.jl is centered around 3 basic tensor operations, i.e. primitives in which every more complicated tensor expression is deconstructed.addition: Add a (possibly scaled version of) one array to another array, where the  indices of the both arrays might appear in different orders. This operation combines  normal array addition and index permutation. It includes as a special case copying one  array into another with permuted indices.\nThe actual implementation is provided by Strided.jl, which contains multithreaded implementations and cache-friendly blocking  strategies for an optimal efficiency.\ntrace or inner contraction: Perform a trace/contraction over pairs of indices of an  array, where the result is a lower-dimensional array. As before, the actual  implementation is provided by Strided.jl.\ncontraction: Performs a general contraction of two tensors, where some indices of  one array are paired with corresponding indices in a second array. This is typically  handled by first permuting (a.k.a. transposing) and reshaping the two input arrays such  that the contraction becomes equivalent to a matrix multiplication, which is then  performed by the highly efficient gemm method from BLAS. The resulting array might  need another reshape and index permutation to bring it in its final form.  Alternatively, a native Julia implementation that does not require the additional  transpositions (yet is typically slower) can be selected by using disable_blas()."
},

{
    "location": "#To-do-list-1",
    "page": "TensorOperations.jl",
    "title": "To do list",
    "category": "section",
    "text": "Make it easier to check contraction order and to splice in runtime information, or   optimize based on memory footprint or other custom cost functions."
},

{
    "location": "indexnotation/#",
    "page": "Index notation with macros",
    "title": "Index notation with macros",
    "category": "page",
    "text": ""
},

{
    "location": "indexnotation/#Index-notation-with-macros-1",
    "page": "Index notation with macros",
    "title": "Index notation with macros",
    "category": "section",
    "text": ""
},

{
    "location": "indexnotation/#The-@tensor-macro-1",
    "page": "Index notation with macros",
    "title": "The @tensor macro",
    "category": "section",
    "text": "The prefered way to specify (a sequence of) tensor operations is by using the @tensor macro, which accepts an index notation format, a.k.a. Einstein notation (and in particular, Einstein\'s summation convention).This can most easily be explained using a simple example:using TensorOperations\nα=randn()\nA=randn(5,5,5,5,5,5)\nB=randn(5,5,5)\nC=randn(5,5,5)\nD=zeros(5,5,5)\n@tensor begin\n    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\n    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]\nendIn the second to last line, the result of the operation will be stored in the preallocated array D, whereas the last line uses a different assignment operator := in order to define a new array E of the correct size. The contents of D and E will be equal.Following Einstein\'s summation convention, the result is computed by first tracing/ contracting the 3rd and 5th index of array A. The resulting array will then be contracted with array B by contracting its 2nd index with the last index of B and its last index with the first index of B. The resulting array has three remaining indices, which correspond to the indices a and c of array A and index b of array B (in that order). To this, the array C (scaled with α) is added, where its first two indices will be permuted to fit with the order a,c,b. The result will then be stored in array D, which requires a second permutation to bring the indices in the requested order a,b,c.In this example, the labels were specified by arbitrary letters or even longer names. Any valid variable name is valid as a label. Note though that these labels are never interpreted as existing Julia variables, but rather are converted into symbols by the @tensor macro. This means, in particular, that the specific tensor operations defined by the code inside the @tensor environment are completely specified at compile time. Alternatively, one can also choose to specify the labels using literal integer constants, such that also the following code specifies the same operation as above. Finally, it is also allowed to use primes (i.e. Julia\'s adjoint operator) to denote different indices, including using multiple subsequent primes.@tensor D[å\'\',ß,c\'] = A[å\'\',1,-3,c\',-3,2]*B[2,ß,1] + α*C[c\',å\'\',ß]The index pattern is analyzed at compile time and expanded to a set of calls to the basic tensor operations, i.e. TensorOperations.add!, TensorOperations.trace! and TensorOperations.contract!. Temporaries are created where necessary, but will by default be saved to a global cache, so that they can be reused upon a next iteration or next call to the function in which the @tensor call is used. When experimenting in the REPL where every tensor expression is only used a single time, it might be better to use disable_cache(), though no real harm comes from using the cache (except higher memory usage). By default, the cache is allowed to take up to the minimum of either one gigabyte or 25% of the total machine memory, though this is fully configurable. We refer to the section on Cache for temporaries for further details.Note that the @tensor specifier can be put in front of a full block of code, or even in front of a function definition, if index expressions are prevalent throughout this block. If a certain part of the code is nonetheless to be interpreted literally and should not be transformed by the @tensor macro, it can be annotated using @notensor, e.g.@tensor function f(args...)\n    some_tensor_expr\n    some_more_tensor_exprs\n    @notensor begin\n        some_literal_indexing_expression\n    end\n    ...\nendNote that @notensor annotations are only needed for indexing expressions which need to be interpreted literally."
},

{
    "location": "indexnotation/#Contraction-order-and-@tensoropt-macro-1",
    "page": "Index notation with macros",
    "title": "Contraction order and @tensoropt macro",
    "category": "section",
    "text": "A contraction of several tensors A[a,b,c,d,e]*B[b,e,f,g]*C[c,f,i,j]*... is generically evaluted as a sequence of pairwise contractions, using Julia\'s default left to right order, i.e. as ( (A[a,b,c,d,e] * B[b,e,f,g]) * C[c,f,i,j]) * ...). Explicit parenthesis can be used to modify this order. Alternatively, if one respects the so-called NCON style of specifying indices, i.e. positive integers for the contracted indices and negative indices for the open indices, the different factors will be reordered and so that the pairwise tensor contractions contract over indices with smaller integer label first. For example,@tensor D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]will be evaluated as (A[-1,3,1,-2,2]*C[1,4,-4,-3])*B[3,2,4,-5]. Furthermore, in that case the indices of the output tensor (D in this case) do not need to be specified (using [:] instead), and will be chosen as (-1,-2,-3,-4,-5). Any other index order for the output tensor is of course still possible by just explicitly specifying it.A final way to enforce a specific order is by giving the @tensor macro a second argument of the form order=(list of indices), e.g.@tensor D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b] order=(f,e,g)This will now first perform the contraction corresponding to the index labeled f, i.e. the contraction between A and C. Then, the contraction corresponding to index labeled e will be performed, which is between B and the result of contracting A and C. If these objects share other contraction indices, in this case g, that contraction will be performed simultaneously, irrespective of its position in the list.Furthermore, there is a @tensoropt macro which will optimize the contraction order to minimize the total number of multiplications (cost model might change or become configurable in the future). The optimal contraction order will be determined at compile time and will be hard coded in the expression resulting from the macro expansion. The cost/size of the different indices can be specified in various ways, and can be integers or some arbitrary polynomial of an abstract variable, e.g. χ. In the latter case, the optimization assumes the assymptotic limit of large χ.@tensoropt D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for all indices (a,b,c,d,e,f)\n@tensoropt (a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost χ for indices a,b,c,e, other indices (d,f) have cost 1\n@tensoropt !(a,b,c,e) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost 1 for indices a,b,c,e, other indices (d,f) have cost χ\n@tensoropt (a=>χ,b=>χ^2,c=>2*χ,e=>5) D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]\n# cost as specified for listed indices, unlisted indices have cost 1 (any symbol for χ can be used)Because of the compile time optimization process, the optimization cannot use run-time information such as the actual sizes of the tensors involved. If these sizes are fixed, they should be hardcoded by specifying the cost in one of the ways as above. The optimization algorithm was described in Physical Review E 90, 033315 (2014) and has a cost that scales exponentially in the number of tensors involved. For reasonably sized tensor network contractions with up to around 30 tensors, this should still be sufficiently fast (at most a few seconds) to be performed once at compile time, i.e. when the contraction is first invoked. Information of the optimization process can be obtained during compilation by using the alternative macro @tensoropt_verbose.The optimal contraction tree as well as the associated cost can be obtained by@optimalcontractiontree D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]where the cost of the indices can be specified in the same various ways as for @tensoropt. In this case, no contraction is performed and the tensors involved do not need to exist."
},

{
    "location": "indexnotation/#Dynamical-tensor-network-contractions-with-ncon-and-@ncon-1",
    "page": "Index notation with macros",
    "title": "Dynamical tensor network contractions with ncon and @ncon",
    "category": "section",
    "text": "Tensor network practicioners are probably more familiar with the network contractor function ncon to perform a tensor network contraction, as e.g. described in NCON. In particular, a graphical application TensorTrace was recently introduced to facilitate the generation of such ncon calls. TensorOperations.jl now provides compatibility with this interface by also exposing an ncon function with the same basic syntaxncon(list_of_tensor_objects, list_of_index_lists)e.g. the example of above is equivalent to@tensor D[:] := A[-1,3,1,-2,2]*B[3,2,4,-5]*C[1,4,-4,-3]\nD ≈ ncon((A,B,C),([-1,3,1,-2,2], [3,2,4,-5], [1,4,-4,-3]))where the lists of tensor objects and of index lists can be given as a vector or a tuple. The ncon function necessarily needs to analyze the contraction pattern at runtime, but this can be an advantage, in case where the contraction is determined by runtime information and thus not known at compile time. A downside from this, besides the fact that this can result in some overhead (though that is typical negligable for anything but very small tensor contractions), is that ncon is type-unstable, i.e. its return type cannot be inferred by the Julia compiler.The full call syntax of the ncon method exposed by TensorOperations.jl isncon(tensorlist, indexlist, [conjlist, sym]; order = ..., output = ...)where the first two arguments are those of above. Let us first discuss the keyword arguments. The keyword argument order can be used to change the contraction order, i.e. by specifying which contraction indices need to be processed first, rather than the strictly increasing order [1,2,...]. The keyword argument output can be used to specify the order of the output indices, when it is different from the default [-1, -2, ...].The optional positional argument conjlist is a list of Bool variables that indicate whether the corresponding tensor needs to be conjugated in the contraction. So whilencon([A,conj(B),C], [[-1,3,1,-2,2], [3,2,4,-5], [1,4,-4,-3]]) ≈\n    ncon([A,B,C], [[-1,3,1,-2,2], [3,2,4,-5], [1,4,-4,-3]], [false, true, false])the latter has the advantage that conjugating B is not an extra step (which creates an additional temporary), but is performed at the same time when it is contracted. The fourth positional argument sym, also optional, can be a constant unique symbol that enables ncon to hook into the global cache structure for storing and recycling temporaries. When it is not specified, the cache cannot be used in any deterministically meaningful way.As an alternative solution to the optional positional arguments, there is also an @ncon macro. It is just a simple wrapper over an ncon call and thus does not analyze the indices at compile time, so that they can be fully dynamical. However, it will transform@ncon([A, conj(B), C], indexlist; order = ..., output = ...)intoncon(Any[A, B, C], indexlist, [false, true, false], some_unique_sym, order = ..., output = ...)so as to get the advantages of cache for temporaries and just-in-time conjugation (pun intended) using the familiar looking ncon syntax.As a proof of principle, let us study the following method for computing the environment to the W isometry in a MERA, as taken from Tensors.net, implemented in three different ways:function IsoEnvW1(hamAB,hamBA,rhoBA,rhoAB,w,v,u)\n    indList1 = Any[[7,8,-1,9],[4,3,-3,2],[7,5,4],[9,10,-2,11],[8,10,5,6],[1,11,2],[1,6,3]]\n    indList2 = Any[[1,2,3,4],[10,7,-3,6],[-1,11,10],[3,4,-2,8],[1,2,11,9],[5,8,6],[5,9,7]]\n    indList3 = Any[[5,7,3,1],[10,9,-3,8],[-1,11,10],[4,3,-2,2],[4,5,11,6],[1,2,8],[7,6,9]]\n    indList4 = Any[[3,7,2,-1],[5,6,4,-3],[2,1,4],[3,1,5],[7,-2,6]]\n    wEnv = ncon(Any[hamAB,rhoBA,conj(w),u,conj(u),v,conj(v)],indList1) +\n   			ncon(Any[hamBA,rhoBA,conj(w),u,conj(u),v,conj(v)],indList2) +\n   			ncon(Any[hamAB,rhoBA,conj(w),u,conj(u),v,conj(v)],indList3) +\n   			ncon(Any[hamBA,rhoAB,v,conj(v),conj(w)],indList4);\n    return wEnv\nend\n\nfunction IsoEnvW2(hamAB,hamBA,rhoBA,rhoAB,w,v,u)\n    indList1 = Any[[7,8,-1,9],[4,3,-3,2],[7,5,4],[9,10,-2,11],[8,10,5,6],[1,11,2],[1,6,3]]\n    indList2 = Any[[1,2,3,4],[10,7,-3,6],[-1,11,10],[3,4,-2,8],[1,2,11,9],[5,8,6],[5,9,7]]\n    indList3 = Any[[5,7,3,1],[10,9,-3,8],[-1,11,10],[4,3,-2,2],[4,5,11,6],[1,2,8],[7,6,9]]\n    indList4 = Any[[3,7,2,-1],[5,6,4,-3],[2,1,4],[3,1,5],[7,-2,6]]\n    wEnv = @ncon(Any[hamAB,rhoBA,conj(w),u,conj(u),v,conj(v)],indList1) +\n   			@ncon(Any[hamBA,rhoBA,conj(w),u,conj(u),v,conj(v)],indList2) +\n   			@ncon(Any[hamAB,rhoBA,conj(w),u,conj(u),v,conj(v)],indList3) +\n   			@ncon(Any[hamBA,rhoAB,v,conj(v),conj(w)],indList4);\n    return wEnv\nend\n\n@tensor function IsoEnvW3(hamAB,hamBA,rhoBA,rhoAB,w,v,u)\n    wEnv[-1,-2,-3] :=\n    	hamAB[7,8,-1,9]*rhoBA[4,3,-3,2]*conj(w[7,5,4])*u[9,10,-2,11]*conj(u[8,10,5,6])*v[1,11,2]*conj(v[1,6,3]) +\n    	hamBA[1,2,3,4]*rhoBA[10,7,-3,6]*conj(w[-1,11,10])*u[3,4,-2,8]*conj(u[1,2,11,9])*v[5,8,6]*conj(v[5,9,7]) +\n    	hamAB[5,7,3,1]*rhoBA[10,9,-3,8]*conj(w[-1,11,10])*u[4,3,-2,2]*conj(u[4,5,11,6])*v[1,2,8]*conj(v[7,6,9]) +\n    	hamBA[3,7,2,-1]*rhoAB[5,6,4,-3]*v[2,1,4]*conj(v[3,1,5])*conj(w[7,-2,6])\n    return wEnv\n    end\nendAll indices appearing in this problem are of size χ. For tensors with ComplexF64 eltype and values of χ in 2:2:32, the reported minimal times using the @belapsed macro from BenchmarkTools.jl are given byχ IsoEnvW1: ncon IsoEnvW2: @ncon IsoEnvW3: @tensor\n2 0.000154413 0.000348091 6.4897e-5\n4 0.000208224 0.000400065 9.5601e-5\n6 0.000558442 0.00076453 0.000354621\n8 0.00138887 0.00150175 0.000982109\n10 0.00506386 0.00365188 0.00288137\n12 0.0126571 0.00959403 0.00818371\n14 0.0292822 0.0216231 0.0184712\n16 0.0531353 0.0410914 0.0359749\n18 0.225333 0.0774705 0.0688475\n20 0.43358 0.139873 0.129315\n22 0.601685 0.243468 0.221995\n24 0.902662 0.459746 0.427615\n26 1.2379 0.66722 0.622856\n28 1.84234 1.08766 1.0322\n30 2.58548 1.53826 1.44854\n32 3.85758 2.44087 2.34229Throughout this range of χ values, method 3 that uses the @tensor macro is consistenly the fastest, both at small χ, where the type stability and the fact that the contraction pattern is analyzed at compile time matters, and at large χ, where the caching of temporaries matters. The direct ncon call has neither of those two features (unless the fourth positional argument is specified, which was not the case here). The @ncon solution provides a hook into the cache and thus is competitive with @tensor for large χ, where the cost is dominated by matrix multiplication and allocations. For small χ, @ncon is also plagued by the runtime analysis of the contraction, but is even worse then ncon. For small χ, the unavoidable type instabilities in ncon implementation seem to make the interaction with the cache hurtful rather than advantageous."
},

{
    "location": "indexnotation/#Multithreading-and-GPU-evaluation-of-tensor-contractions-with-@cutensor-1",
    "page": "Index notation with macros",
    "title": "Multithreading and GPU evaluation of tensor contractions with @cutensor",
    "category": "section",
    "text": "Every index expression will be evaluated as a sequence of elementary tensor operations, i.e. permuted additions, partial traces and contractions, which are implemented for strided arrays as discussed in Package Features. In particular, these implementations rely on Strided.jl, and we refer to this package for a full specification of which arrays are supported. As a rule of thumb, Arrays from Julia base, as well as views thereof if sliced with a combination of Integers and Ranges. Special types such as Adjoint and Transpose from Base are also supported. For permuted addition and partial traces, native Julia implementations are used which could benefit from multithreading if JULIA_NUM_THREADS>1. The binary contraction is performed by first permuting the two input tensors into a form such that the contraction becomes equivalent to one matrix multiplication on the whole data, followed by a final permutation to bring the indices of the output tensor into the desired order. This approach allows to use the highly efficient matrix multiplication (gemm) from BLAS, which is multithreaded by default. There is also a native contraction implementation that is used for e.g. arrays with an eltype that is not <:LinearAlgebra.BlasFloat. It performs the contraction directly without the additional permutations, but still in a cache-friendly and multithreaded way (again relying on JULIA_NUM_THREADS>1). This implementation can sometimes be faster even for BlasFloat types, and the use of BLAS can be disabled globally by calling disable_blas(). It is currently not possible to control the use of BLAS at the level of individual contractions.Since TensorOperations v2.0, the necessary implementations are also available for CuArray objects of the CuArrays.jl library. This implementation is essentially a simple wrapper over the CUTENSOR library of NVidia, and as such has certain restrictions as a result thereof. Native Julia alternatives using CUDAnative might be provided in the future.Mixed operations between host arrays (e.g. Array) and device arrays (e.g. CuArray) will fail. However, if one wants to harness the computing power of the GPU to perform all tensor operations, there is a dedicated macro @cutensor. This will transfer all arrays to the GPU before performing the requested operations. If the output is an existing host array, the result will be copied back. If a new result array is created (i.e. using :=), it will remain on the GPU device and it is up to the user to transfer it back. Arrays are transfered to the GPU just before they are first used, and in a complicated tensor expression, this might have the benefit that transer of the later arrays overlaps with computation of earlier operations."
},

{
    "location": "functions/#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions/#TensorOperations.tensorcopy!",
    "page": "Functions",
    "title": "TensorOperations.tensorcopy!",
    "category": "function",
    "text": "tensorcopy!(A, IA, C, IC)\n\nCopies A into C by permuting the dimensions according to the pattern specified by IA and IC. Both iterables should contain the same elements in a different order. The result of this method is equivalent to permutedims!(C, A, p) where p is the permutation such that IC=IA[p]. The implementation of tensorcopy! is however more efficient on average, especially if Threads.nthreads() > 1.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensoradd!",
    "page": "Functions",
    "title": "TensorOperations.tensoradd!",
    "category": "function",
    "text": "tensoradd!(α, A, IA, β, C, IC)\n\nUpdates C to β*C + α * tensorcopy(A,IA,IC), but without creating the temporary permuted array.\n\nSee also: tensorcopy\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensortrace!",
    "page": "Functions",
    "title": "TensorOperations.tensortrace!",
    "category": "function",
    "text": "tensortrace!(α, A, IA, β, C, IC)\n\nUpdates C to β*C + α tensortrace(A,IA,IC), but without creating the temporary traced array.\n\nSee also: tensortrace\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcontract!",
    "page": "Functions",
    "title": "TensorOperations.tensorcontract!",
    "category": "function",
    "text": "tensorcontract!(α, A, labelsA, conjA, B, labelsB, conjB, β, C, labelsC)\n\nReplaces C with β C + α A * B, where some indices of array A are contracted with corresponding indices in array B by assigning them identical labels in the iterables labelsA and labelsB. The arguments conjA and conjB should be of type Char and indicate whether the data of arrays A and B, respectively, need to be conjugated (value \'C\') or not (value \'N\'). Every label should appear exactly twice in the union of labelsA, labelsB and labelsC, either in the intersection of labelsA and labelsB (for indices that need to be contracted) or in the interaction of either labelsA or labelsB with labelsC, for indicating the order in which the open indices should be match to the indices of the output array C.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorproduct!",
    "page": "Functions",
    "title": "TensorOperations.tensorproduct!",
    "category": "function",
    "text": "tensorproduct!(α, A, labelsA, B, labelsB, β, C, labelsC)\n\nReplaces C with β C + α A * B without any indices being contracted.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcopy",
    "page": "Functions",
    "title": "TensorOperations.tensorcopy",
    "category": "function",
    "text": "tensorcopy(A, IA, IC = IA)\n\nCreates a copy of A, where the dimensions of A are assigned indices from the iterable IA and the indices of the copy are contained in IC. Both iterables should contain the same elements in a different order.\n\nThe result of this method is equivalent to permutedims(A, p) where p is the permutation such that IC = IA[p]. The implementation of tensorcopy is however more efficient on average, especially if Threads.nthreads() > 1.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensoradd",
    "page": "Functions",
    "title": "TensorOperations.tensoradd",
    "category": "function",
    "text": "tensoradd(A, IA, B, IB, IC = IA)\n\nReturns the result of adding arrays A and B where the iterabels IA and IB denote how the array data should be permuted in order to be added. More specifically, the result of this method is equivalent to\n\ntensorcopy(A, IA, IC) + tensorcopy(B, IB, IC)\n\nbut without creating the temporary permuted arrays.\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensortrace",
    "page": "Functions",
    "title": "TensorOperations.tensortrace",
    "category": "function",
    "text": "tensortrace(A, IA [, IC])\n\nTrace or contract pairs of indices of array A, by assigning them an identical indices in the iterable IA. The untraced indices, which are assigned a unique index, can be reordered according to the optional argument IC. The default value corresponds to the order in which they appear. Note that only pairs of indices can be contracted, so that every index in IA can appear only once (for an untraced index) or twice (for an index in a contracted pair).\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorcontract",
    "page": "Functions",
    "title": "TensorOperations.tensorcontract",
    "category": "function",
    "text": "tensorcontract(A, IA, B, IB[, IC])\n\nContract indices of array A with corresponding indices in array B by assigning them identical labels in the iterables IA and IB. The indices of the resulting array correspond to the indices that only appear in either IA or IB and can be ordered by specifying the optional argument IC. The default is to have all open indices of array A followed by all open indices of array B. Note that inner contractions of an array should be handled first with tensortrace, so that every label can appear only once in IA or IB seperately, and once (for open index) or twice (for contracted index) in the union of IA and IB.\n\nThe contraction can be performed by a native Julia algorithm without creating any temporaries, or by first permuting the arrays such that the contraction becomes equivalent to a matrix product, which is then performed by BLAS. The latter is typically faster for large arrays. The choice of method is globally controlled by the methods enable_blas() and disable_blas().\n\n\n\n\n\n"
},

{
    "location": "functions/#TensorOperations.tensorproduct",
    "page": "Functions",
    "title": "TensorOperations.tensorproduct",
    "category": "function",
    "text": "tensorproduct(A, IA, B, IB, IC = (IA..., IB...))\n\nComputes the tensor product of two arrays A and B, i.e. returns a new array C with ndims(C) = ndims(A)+ndims(B). The indices of the output tensor are related to those of the input tensors by the pattern specified by the indices. Essentially, this is a special case of tensorcontract with no indices being contracted over. This method checks whether the indices indeed specify a tensor product instead of a genuine contraction.\n\n\n\n\n\n"
},

{
    "location": "functions/#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "The elementary tensor operations can also be accessed via functions, mainly for compatibility with older versions of this toolbox. The function-based syntax is also required when the contraction pattern is not known at compile time but is rather determined dynamically.These functions come in a mutating and non-mutating version. The mutating versions mimick the argument order of some of the BLAS functions, such as blascopy!, axpy! and gemm!. Symbols A and B always refer to input arrays, whereas C is used to denote the array where the result will be stored. They also return C and are therefore type stable. The greek letters α and β denote scalar coefficients.tensorcopy!\ntensoradd!\ntensortrace!\ntensorcontract!\ntensorproduct!The non-mutating functions are simpler in not allowing scalar coefficients and conjugation. They also take a default value for the labels of the output array if these are not specified. However, the return type is only inferred if the labels are entered as tuples, and also IC is specified. They are simply called as:tensorcopy\ntensoradd\ntensortrace\ntensorcontract\ntensorproduct"
},

{
    "location": "cache/#",
    "page": "Cache for temporaries",
    "title": "Cache for temporaries",
    "category": "page",
    "text": ""
},

{
    "location": "cache/#Cache-for-temporaries-1",
    "page": "Cache for temporaries",
    "title": "Cache for temporaries",
    "category": "section",
    "text": "Contracting a sequence of tensors is provably most efficient (in terms of number of computations) by contracting them pairwise. However, this requires that several intermediate results need to be stored. In addition, if the contraction needs to be performed as a BLAS matrix multiplication (which is typically the fastest choice), every tensor typically needs an additional permuted copy that is compatible with the implementation of the contraction as multiplication. All these temporary arrays, which can be large, put a a lot of pressure on Julia\'s garbage collector, and the total time spent in the garbage collector can become significant.That\'s why there is now a functionality to store intermediate results in a package wide cache, where they can be reused upon a next run, either a next iteration if the tensor contraction appears within the body of a loop, or on the next function call if it appears directly within a given function. This mechanism only works with the @tensor macro, not with the function-based interface.The @tensor macro expands the given expression and immediately generates the code to create the necessary temporaries. It associates with each of them a random symbol (gensym()) and uses this as an identifier (together with the Threads.threadid of where it it is being evaluated) in a package wide global cache structure TensorOperations.cache, the implementation of which is a least-recently used cache dictionary from LRUCache.jl. Thereto, it estimates the size of each object added to the cache using Base.summarysize and discards objects once a certain memory limit is reached."
},

{
    "location": "cache/#TensorOperations.enable_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.enable_cache",
    "category": "function",
    "text": "enable_cache(; maxsize::Int = ..., maxrelsize::Real = ...)\n\n(Re)-enable the cache for further use; set the maximal size maxsize (as number of bytes) or relative size maxrelsize, as a fraction between 0 and 1, resulting in maxsize = floor(Int, maxrelsize * Sys.total_memory()). Default value is maxsize = 2^30 bytes, which amounts to 1 gigabyte of memory.\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.disable_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.disable_cache",
    "category": "function",
    "text": "disable_cache()\n\nDisable the cache for further use but does not clear its current contents. Also see clear_cache()\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.cachesize",
    "page": "Cache for temporaries",
    "title": "TensorOperations.cachesize",
    "category": "function",
    "text": "cachesize()\n\nReturn the current memory size (in bytes) of all the objects in the cache.\n\n\n\n\n\n"
},

{
    "location": "cache/#TensorOperations.clear_cache",
    "page": "Cache for temporaries",
    "title": "TensorOperations.clear_cache",
    "category": "function",
    "text": "clear_cache()\n\nClear the current contents of the cache.\n\n\n\n\n\n"
},

{
    "location": "cache/#Enabling-and-disabling-the-cache-1",
    "page": "Cache for temporaries",
    "title": "Enabling and disabling the cache",
    "category": "section",
    "text": "The use of the cache can be enabled or disabled usingenable_cache\ndisable_cacheFurthermore, the current total size of all the objects stored in the cache can be obtained using the method cachesize, and clear_cache can be triggered to release all the objects currently stored in the cache, such that they can be removed by Julia\'s garbage collector.cachesize\nclear_cache"
},

{
    "location": "cache/#Cache-and-multithreading-1",
    "page": "Cache for temporaries",
    "title": "Cache and multithreading",
    "category": "section",
    "text": "The LRU cache currently is thread safe, but requires that temporary objects are allocated for every thread that is running @tensor expressions. If the same tensor contraction is evaluated by several threads (simultaneoulsy or not, this we cannot know), every thread needs to have its own set of temporary variables, so the \'same\' temporary will be stored in the cache multiple times. As indicated above, this is accomplished by associating with every temporary a randoms symbol and the Threads.threadid of the thread where the expression is being evaluated. If you have JULIA_NUM_THREADS>1 but always run tensor expressions on the main execution thread, no additional copies of the temporaries will be created."
},

{
    "location": "implementation/#",
    "page": "Implementation",
    "title": "Implementation",
    "category": "page",
    "text": ""
},

{
    "location": "implementation/#Implementation-1",
    "page": "Implementation",
    "title": "Implementation",
    "category": "section",
    "text": "** Warning: this section still needs to be updated for version 2.0 **"
},

{
    "location": "implementation/#Index-notation-and-the-@tensor-macro-1",
    "page": "Implementation",
    "title": "Index notation and the @tensor macro",
    "category": "section",
    "text": "We start by describing the implementation of the @tensor and @tensoropt macro. The macros end up transforming the index expression to the corresponding function calls to the primitive building blocks, which are discussed in the next section. In principle, anyone interested in making the @tensor macro work for custom array types should only reimplement these building blocks, but it is useful to understand how the tensor expressions are processed."
},

{
    "location": "implementation/#Tensors,-a.k.a-indexed-objects-1",
    "page": "Implementation",
    "title": "Tensors, a.k.a indexed objects",
    "category": "section",
    "text": "The central objects in tensor expressions that follow the @tensor macro are the index expressions of the form A[a,b,c].  These are detected as subexpressions of the form Expr(:ref, args). In fact, what is recognized by @tensor as an indexed object is more general, and also includes expressions of the form A[a b c] or A[a b; c d e] (Expr(:typed_hcat, args) and Expr(:typed_vcat, args)). The last form in particular is useful in more general contexts, and allows to distinguish between two sets of indices, referred to as left (a and b) and right (c, d and e) indices. This can be used when the @tensor macro wants to be generalized to user types, which for example distinguish between contravariant (upper) and covariant (lower) indices. For AbstractArray subtypes, such distinction is of course meaningless.Note that the object being indexed, i.e. A in all of the above examples or args[1] in the corresponding Expr objects, can itself be an expression, that is not further analyzed. In particular, this may itself contain further indexing objects, such that one can get tensor objects from a list, e.g. list[3][a,b,c] or slice an array before using it in the @tensor expression, e.g. A[1:2:end,:,3:end][a,b,c].Everything appearing in the [ ], e.g. args[2:end] (in case of :ref or :typed_hcat, the argument structure of :typed_vcat is slightly more complicated) is considered to be a valid index. This can be any valid Julia variable name, which is just kept as symbol, or any literal integer constant, (for legacy reasons, any literal character constant,) or finally an Expr(Symbol(\"\'\"),args), i.e. an expression of the form :(a\'). The latter is converted to a symbol of the form Symbol(\"a′\") when a is itself a symbol or integer, or this is applied recursively if a contains more primes.The implementation for detecting tensors and indices (istensor, isindex) and actually converting them to a useful format (maketensor, makeindex) are found in src/indexnotation/tensorexpressions.jl. In particular, maketensor will return the indexed object, which is just esc(args[1]), the list of left indices and the list of right indices.Furthermore, there is isscalar and makescalar to detect and process subexpressions that will evaluate to a scalar. Finally, there is isgeneraltensor and makegeneraltensor to detect and process a a tensor (indexed object), that is possibly conjugated or multiplied with a scalar. This is useful because the primitive tensor operations (i.e. see Building blocks below), accept a scalar factor and conjugation flag, so that these operations can be done simultaneously and do not need to be evaluated at first (which would require additional temporaries). The function makegeneraltensor in particular will return the indexed object, the list of left indices, the list of right indices, the scalar factor, and a flag (Bool) that indicates whether the object needs to be conjugated (true) or not (false).The file src/indexnotation/tensorexpressions.jl also contains simple methods to detect assignment (isassignment) into existing objects (i.e. =, += and -=) or so-called definitions (isdefinition), that create a new object (via := or its Unicode variant ≔, obtained as \\coloneq + TAB). The function getlhsrhs will return the left hand side and right hand side of an assignment or definition expression separately.Finally, there are methods to detect whether the right hand side is a valid tensor expression (istensorexpr) and to get the indices of a complete tensor expressions. In particular,  getindices returns a list of indices that will remain after the expression is evaluated (i.e. any index that is not contracted in the expression because it only appears once), whereas getallindices returns a list of all indices that appear in the expression. The latter is used to analyze complete tensor contraction graphs."
},

{
    "location": "implementation/#The-macros-@tensor-and-@tensoropt-1",
    "page": "Implementation",
    "title": "The macros @tensor and @tensoropt",
    "category": "section",
    "text": "Actual processing of the complete expression that follows the @tensor macro and converting it into a list of actual calls to the primitive tensor operations is handled by the functions defined in src/indexnotation/tensormacro.jl. The integral expression received by the @tensor macro is passed on to the tensorify function. The @tensoropt macro will first generate the data required to optimize contraction order, by calling optdata. If no actual costs are specified, i.e. @tensoropt receives a single expression, then optdata just assigns the same cost to all indices in the expression. Otherwise, the expression that specifies the costs need to be parsed first (parsecost). Finally, @tensoropt also calls tensorify passing the optdata as a second optional argument (whose default value nothing is used by @tensor).The function tensorify consists of several steps. Firstly, it canonicalizes the expression. Currently, this involves a single pass which expands all conj calls of e.g. products or sums to a conj call on each of the arguments (via the expandconj function). Secondly, tensorify will process the contraction order using processcontractorder. This starts from the fact that a product of several tensors, specified as A[...]*B[...]*C[...] yields a single Expr(:call,[:*,...]) object where all the factors are still together. If a user wants a specific order, he can do so by grouping them with parenthesis. Whenever an expression Expr(:call,[:*,...]) is found where more than two of the arguments satisfy isgeneraltensor, processcontractorder will convert it into a nested set of pairwise multiplications according to a number of strategies discussed in the next subsection.The major part of tensorify is to generate the correct function calls corresponding to the tensor expression. It detects assignments or definitions (the most common case) and validates the left hand side thereof (i.e. it should satisfy istensor and have no duplicate indices). Then, it generates the corresponding function calls corresponding to the index expression by passing onto the deindexify function, which takes the signature julia deindexify(dst, β, ex, α, leftind, rightind, istemporary = false) Here, dst is the symbol or expression corresponding to the destination object; it\'s nothing in case of a definition (:=), i.e. if the object corresponding to the result needs to be created/allocated. β and α are isscalar expressions, and deindexify will create the function calls required to update dst with multiplying dst with β and adding α times the result of the expression ex to it; this is supported as a one step process by each of the primitive operations. leftind and rightind correspond to the list of indices of the left hand side of the defition or assignment. The final argument istemporary indicates, if dst == nothing and a new object needs to be created/allocated, whether it is a temporary object. If istemporary == true, it can be stored in the cache and later retrieved. If istemporary == false, it corresponds to an explicit left hand side created by the user in a definition, and should not be in the cache.The function deindexify will determine the top level operation represented by ex (which should be a istensorexpr), and then pass on to  deindexify_generaltensor, deindexify_linearcombination, deindexify_contraction for actually creating the correct function call expressions. If any of the arguments of e.g. a linear combination or a tensor contraction is itself a composite tensor expression (i.e. not a isgeneraltensor), deindexify is called recursively."
},

{
    "location": "implementation/#Analyzing-contraction-graphs-(a.k.a-tensor-networks)-and-optimizing-contraction-order-1",
    "page": "Implementation",
    "title": "Analyzing contraction graphs (a.k.a tensor networks) and optimizing contraction order",
    "category": "section",
    "text": "The function processcontractorder, which is excuted before the index expression is converted to function calls, will detect any multiplication with more than two isgeneraltensor factors, and divide it up into a nested sequence of pairwise multiplications (tensor contractions), i.e. a tree. If the @tensor macro was used, optdata = nothing and in principle the multiplication will be performed from left to right. There is one exception, which is that if the indices follow the NCON convention, i.e. negative integers are used for uncontracted indices and positive integers for contracted indices. Then the contraction tree is built such that tensors that share the contraction index which is the lowest positive integer are contracted first. Relevant code can be found in src/indexnotation/ncontree.jlWhen the @tensoropt macro was used, optdata is a dictionary associating a cost (either a number or a polynomial in some abstract scaling parameter) to every index, and this information is used to determine the (asymptotically) optimal contraction tree (in terms of number of floating point operations). The code for the latter is in src/indexnotation/optimaltree.jl, with the lightweight polynomial implementation in src/indexnotation/polynomial.jl. Aside from a generic polynomial type Poly, the latter also contains a Power type which represents a single term of a polynomial (i.e. a scalar coefficient and an exponent). This type is closed under multiplication, and can be multiplied much more efficiently. Only under addition is a generic Poly returned."
},

{
    "location": "implementation/#TensorOperations.add!",
    "page": "Implementation",
    "title": "TensorOperations.add!",
    "category": "function",
    "text": "add!(α, A, conjA, β, C, indleft, indright)\n\nImplements C = β*C+α*permute(op(A)) where A is permuted such that the left (right) indices of C correspond to the indices indleft (indright) of A, and op is conj if conjA == :C or the identity map if conjA == :N (default). Together, (indleft..., indright...) is a permutation of 1 to the number of indices (dimensions) of A.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.trace!",
    "page": "Implementation",
    "title": "TensorOperations.trace!",
    "category": "function",
    "text": "trace!(α, A, conjA, β, C, indleft, indright, cind1, cind2)\n\nImplements C = β*C+α*partialtrace(op(A)) where A is permuted and partially traced, such that the left (right) indices of C correspond to the indices indleft (indright) of A, and indices cindA1 are contracted with indices cindA2. Furthermore, op is conj if conjA == :C or the identity map if conjA=:N (default). Together, (indleft..., indright..., cind1, cind2) is a permutation of 1 to the number of indices (dimensions) of A.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.contract!",
    "page": "Implementation",
    "title": "TensorOperations.contract!",
    "category": "function",
    "text": "contract!(α, A, conjA, B, conjB, β, C, oindA, cindA, oindB, cindB, indleft, indright, syms = nothing)\n\nImplements C = β*C+α*contract(opA(A),opB(B)) where A and B are contracted, such that the indices cindA of A are contracted with indices cindB of B. The open indices oindA of A and oindB of B are permuted such that C has left (right) indices corresponding to indices indleft (indright) out of (oindA..., oindB...). The operation opA (opB) acts as conj if conjA (conjB) equal :C or as the identity map if conjA (conjB) equal :N. Together, (oindA..., cindA...) is a permutation of 1 to the number of indices of A and (oindB..., cindB...) is a permutation of 1 to the number of indices of C. Furthermore, length(cindA) == length(cindB), length(oindA)+length(oindB) equals the number of indices of C and (indleft..., indright...) is a permutation of 1 ot the number of indices of C.\n\nThe final argument syms is optional and can be either nothing, or a tuple of three symbols, which are used to identify temporary objects in the cache to be used for permuting A, B and C so as to perform the contraction as a matrix multiplication.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.checked_similar_from_indices",
    "page": "Implementation",
    "title": "TensorOperations.checked_similar_from_indices",
    "category": "function",
    "text": "checked_similar_from_indices(C, T, indleft, indright, A, conjA = :N)\n\nReturns an object similar to A which has an eltype given by T and whose left indices correspond to the indices indleft from op(A), and its right indices correspond to the indices indright from op(A), where op is conj if conjA == :C or does nothing if conjA == :N (default). Here, C is a potential candidate for the similar object. If C === nothing, or its its eltype or shape does not match, a new object is allocated and returned. Otherwise, C is returned.\n\n\n\n\n\nchecked_similar_from_indices(C, T, indoA, indoB, indleft, indright,\n                                A, B, conjA = :N, conjB= :N)\n\nReturns an object similar to A which has an eltype given by T and dimensions/sizes corresponding to a selection of those of opA(A) and opB(B) concatenated. Out of the collection of indices in indoA of opA(A) and indoB of opB(B), we construct an object whose left (right) indices correspond to indices indleft (indright) from that collection. Furthermore, C is a potential candidate for the similar object. If C === nothing, or its its eltype or shape does not match, a new object is allocated and returned. Otherwise, C is returned.\n\n\n\n\n\n"
},

{
    "location": "implementation/#TensorOperations.scalar",
    "page": "Implementation",
    "title": "TensorOperations.scalar",
    "category": "function",
    "text": "scalar(C)\n\nReturns the single element of a tensor-like object with zero indices or dimensions.\n\n\n\n\n\n"
},

{
    "location": "implementation/#Building-blocks-1",
    "page": "Implementation",
    "title": "Building blocks",
    "category": "section",
    "text": "The @tensor macro converts the index expression into a set of function calls corresponding to three primitive operations: addition, tracing and contraction. These operations are implemented for arbitrary strided arrays from Julia Base, i.e. Arrays, views with ranges thereof, and certain reshape operations. This includes certain arrays that can only be determined to be strided on runtime, and does therefore not coincide with the type union StridedArray from Julia Base. In fact, the methods accept AbstractArray objects, but convert these to (Unsafe)StridedView objects from the package Strided.jl, and we refer to this package for a more detailed discussion on which arrays are supported and why.The primitive tensor operations are captured by the following mutating methods (note that these are not exported)TensorOperations.add!\nTensorOperations.trace!\nTensorOperations.contract!These are the central objects that should be overloaded by custom tensor types that would like to be used within the @tensor environment. They are also used by the function based methods discussed in the section Functions.Furthermore, it is essential to be able to construct new tensor objects that are similar to existing ones, i.e. to place the result of the computation in case no output is specified. In order to reuse temporary objects stored in the global cache, this method also receives a candidate similar object, which it can return if it matches the requirements.TensorOperations.checked_similar_from_indicesNote that the type of the cached object is not known to the compiler, as the cache stores objects as Any. Therefore, the function checked_similar_from_indices should try to restore the type information. By passing any object retrieved from the cache through this function, type stability within the @tensor macro can then still be guaranteed.Finally, there is a particularly simple method scalar whose sole purpose is to extract the single entry of an object with zero indices, i.e. an instance of AbstractArray{T,0} in case of Julia Base arrays:TensorOperations.scalarThe implementation of all of these methods can be found in src/implementation/stridedarray.jl.By implementing these five methods for other types that represent some kind of tensor or multidimensional object, they can be used in combination with the @tensor macro. In particular, we also provide basic support for contracting a Diagonal matrix with an arbitrary strided array in src/implementation/diagonal.jl."
},

]}
