using Documenter
using TensorOperations

makedocs(modules=[TensorOperations],
            sitename = "TensorOperations.jl",
            authors = "Jutho Haegeman",
            format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
            pages = [
                "Home" => ["index.md",
                            "indexnotation.md",
                            "functions.md",
                            "cache.md",
                            "implementation.md"]
            ])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/Jutho/TensorOperations.jl.git")
