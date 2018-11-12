using Documenter
using TensorOperations

makedocs(modules=[TensorOperations],
            format=:html,
            sitename="TensorOperations.jl",
            pages = [
                "Home" => ["index.md", "indexnotation.md", "functions.md", "cache.md", "implementation.md"]
            ])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    deps = nothing,
    make = nothing,
    target = "build",
    repo = "github.com/Jutho/TensorOperations.jl.git",
)
